"""
ColorSAR - conditional pixel-space diffusion: SAR (s1) -> optical (s2) in Lab

A Palette-style conditional DDPM. The network sees the SAR tile at three despeckling
levels (raw, Gaussian sigma 1, sigma 2.5 - the same "sar3" idea as Model 3) and
generates all three Lab channels of the optical tile (brightness L and color a/b)
directly at full 256x256 resolution. Unlike the regression/classification colorizers,
it does not average uncertain outcomes into a blur: every sample is one plausible,
sharp image.

    python train_diffusion.py --root v_2 --hours 7          # the real run (GPU)
    python train_diffusion.py --root v_2 --smoke             # tiny CPU check of every code path

Split: whole Sentinel scenes are held out (scene_split.json: the same val/test scenes
as the team benchmark), so test tiles come from places the model never saw.

Training
    eps-prediction, cosine noise schedule (T=1000), AdamW, EMA of the weights, AMP,
    DataParallel on several GPUs, rot90/flip augmentation. The time budget (--hours)
    is wall time. For the first --full_frac of it the model trains on random 128x128
    crops (4x cheaper per sample); the rest is on full 256x256 tiles, the resolution it
    is sampled at.

Evaluation (on the held-out test scenes, full 256 px)
    DDIM sampling with --ddim_steps steps. "sample" = one draw per tile (what a user
    sees); "mean of K" = the average of K draws (closer to the most likely image, lower
    RMSE, but blurrier). Reported per row: RGB PSNR/SSIM (as evaluate.py), a/b RMSE and
    L RMSE in Lab units, sharpness (variance of the Laplacian, as _sharpness.py), LPIPS
    (if the lpips package is installed). Baselines use the model's own L with
    a = b = 0 ("no color") or the mean training color ("mean color").

Outputs in --out (default diffusion_run/):
    model_diffusion.pth  EMA weights (fp16) + meta      history.json   loss / val log
    metrics_test.json    all metrics, overall + per terrain
    sample_predictions.png, loss_curve.png, split.json
"""
import argparse
import copy
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from color_utils import _RGB_TO_XYZ, _WHITE_D65, _f_l, _srgb_to_linear, find_pairs, lab_to_rgb
from pipeline import _gauss

AB_SCALE = 60.0     # a/b in Lab units / AB_SCALE -> network range (satellite colors are |a|,|b| < 60)
_S1_NAME = re.compile(r"^(?P<roi>.+)_s1_(?P<scene>\d+)_p\d+\.png$")


# =========================================================================== data
def scene_of(sar_path):
    m = _S1_NAME.match(os.path.basename(sar_path))
    if m is None:
        raise ValueError(f"cannot read a scene id from {sar_path!r}")
    return f"{m['roi']}_{m['scene']}"


def scene_split(pairs, split_file):
    s = json.load(open(split_file))
    val, test = set(s["val"]), set(s["test"])
    parts = {"train": [], "val": [], "test": []}
    for p in pairs:
        sc = scene_of(p[0])
        parts["test" if sc in test else "val" if sc in val else "train"].append(p)
    return parts


def subsample(pairs, n, seed):
    if not n or n >= len(pairs):
        return pairs
    idx = np.random.RandomState(seed).choice(len(pairs), n, replace=False)
    return [pairs[i] for i in sorted(idx)]


class Part:
    """A split held in memory as uint8: sar (N,256,256), rgb (N,256,256,3)."""

    def __init__(self, pairs, workers=8):
        def read(p):
            s = np.array(Image.open(p[0]).convert("L"))
            o = np.array(Image.open(p[1]).convert("RGB"))
            if s.shape != (256, 256):
                s = cv2.resize(s, (256, 256), interpolation=cv2.INTER_AREA)
            if o.shape[:2] != (256, 256):
                o = cv2.resize(o, (256, 256), interpolation=cv2.INTER_AREA)
            return s, o
        with ThreadPoolExecutor(workers) as ex:
            out = list(ex.map(read, pairs))
        self.pairs = pairs
        self.terrain = [p[2] for p in pairs]
        self.sar = np.stack([o[0] for o in out]) if out else np.zeros((0, 256, 256), np.uint8)
        self.rgb = np.stack([o[1] for o in out]) if out else np.zeros((0, 256, 256, 3), np.uint8)

    def __len__(self):
        return len(self.pairs)

    def tensors(self, idx, dev):
        g = torch.from_numpy(self.sar[idx]).to(dev).float().div_(255).unsqueeze(1)
        rgb = torch.from_numpy(self.rgb[idx]).to(dev).permute(0, 3, 1, 2).float().div_(255)
        return g, rgb


# =========================================================================== Lab
def rgb_to_lab_t(rgb):
    """(N,3,H,W) RGB in [0,1] -> (L in [0,100], a, b) in Lab units, as skimage.rgb2lab."""
    lin = _srgb_to_linear(rgb.clamp(0.0, 1.0))
    m = torch.tensor(_RGB_TO_XYZ, dtype=lin.dtype, device=lin.device)
    xyz = torch.einsum("ij,njhw->nihw", m, lin)
    w = torch.tensor(_WHITE_D65, dtype=lin.dtype, device=lin.device).view(1, 3, 1, 1)
    f = _f_l((xyz / w).clamp_min(0.0))
    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], 1)


def encode(rgb):
    """RGB [0,1] -> network target (N,3,H,W) in about [-1,1]: L/50-1, a/AB_SCALE, b/AB_SCALE."""
    lab = rgb_to_lab_t(rgb)
    return torch.cat([lab[:, :1] / 50.0 - 1.0, lab[:, 1:] / AB_SCALE], 1)


def decode_lab(x):
    """Network output -> Lab units (L, a, b)."""
    x = x.float()
    return torch.cat([(x[:, :1] + 1.0) * 50.0, x[:, 1:] * AB_SCALE], 1)


def lab_to_rgb_np(lab):
    """Lab units (N,3,H,W) -> (N,H,W,3) RGB in [0,1] via color_utils.lab_to_rgb (skimage),
    i.e. exactly the conversion evaluate.py scores with."""
    L = lab[:, :1] / 50.0 - 1.0
    ab = 2.0 * (lab[:, 1:] + 128.0) / 255.0 - 1.0
    return lab_to_rgb(L, ab)


def condition(gray):
    """SAR gray (N,1,H,W) in [0,1] -> (N,3,H,W) in [-1,1]: raw, sigma 1, sigma 2.5."""
    return torch.cat([gray, _gauss(gray, 1.0), _gauss(gray, 2.5)], 1) * 2.0 - 1.0


def augment(*xs):
    """The same random rot90 + flip for every tensor (tiles have no 'up')."""
    k, flip = np.random.randint(4), np.random.rand() < 0.5
    out = []
    for x in xs:
        x = torch.rot90(x, k, dims=(2, 3))
        out.append(torch.flip(x, dims=(3,)) if flip else x)
    return out


def random_crop(xs, size):
    if size >= xs[0].shape[-1]:
        return xs
    n, H = xs[0].shape[0], xs[0].shape[-1]
    ys = np.random.randint(0, H - size + 1, n)
    xs_ = np.random.randint(0, H - size + 1, n)
    return [torch.stack([x[i, :, ys[i]:ys[i] + size, xs_[i]:xs_[i] + size] for i in range(n)])
            for x in xs]


# =========================================================================== network
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], 1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, 2 * cout)
        self.n2, self.drop = nn.GroupNorm(32, cout), nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(F.silu(temb))[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + scale) + shift
        return self.skip(x) + self.c2(self.drop(F.silu(h)))


class Attention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads, self.norm = heads, nn.GroupNorm(32, c)
        self.qkv, self.proj = nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(n, 3, self.heads, c // self.heads, h * w).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(n, c, h, w))


class DiffusionUNet(nn.Module):
    """eps-prediction U-Net over [noisy Lab (3) | SAR condition (3)]. Attention only at the
    lowest resolution (16x16 for 128 crops, 32x32 for full tiles) and in the middle."""

    def __init__(self, ch=64, mult=(1, 2, 3, 4), n_res=2, dropout=0.1, in_ch=6, out_ch=3):
        super().__init__()
        tdim, self.ch = ch * 4, ch
        self.temb = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        last = len(mult) - 1
        self.down, skips, c = nn.ModuleList(), [ch], ch
        for lvl, m in enumerate(mult):
            for _ in range(n_res):
                blk = nn.ModuleList([ResBlock(c, ch * m, tdim, dropout)])
                c = ch * m
                if lvl == last:
                    blk.append(Attention(c))
                self.down.append(blk)
                skips.append(c)
            if lvl < last:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                skips.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, tdim, dropout), Attention(c), ResBlock(c, c, tdim, dropout)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(mult))):
            for i in range(n_res + 1):
                blk = nn.ModuleList([ResBlock(c + skips.pop(), ch * m, tdim, dropout)])
                c = ch * m
                if lvl == last:
                    blk.append(Attention(c))
                if lvl > 0 and i == n_res:
                    blk.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blk.append(nn.Conv2d(c, c, 3, padding=1))
                self.up.append(blk)
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _run(blk, h, temb):
        for layer in blk:
            h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
        return h

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.ch))
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            h = self._run(blk, h, temb)
            hs.append(h)
        h = self._run(self.mid, h, temb)
        for blk in self.up:
            h = self._run(blk, torch.cat([h, hs.pop()], 1), temb)
        return self.out(h)


class Diffusion:
    """Cosine-schedule DDPM (Nichol & Dhariwal 2021) with a deterministic DDIM sampler."""

    def __init__(self, T=1000, device="cpu"):
        self.T = T
        s = 0.008
        x = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((x + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        self.abar = torch.cumprod(1 - betas, 0).float().to(device)

    def q_sample(self, x0, t, noise):
        a = self.abar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, cond, steps, amp, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn((cond.shape[0], 3) + tuple(cond.shape[-2:]), generator=g).to(cond.device)
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        for i, t in enumerate(ts):
            tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            with torch.autocast("cuda", enabled=amp):
                eps = model(torch.cat([x, cond], 1), tt).float()
            a = self.abar[t]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            a_next = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return x


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def ema_update(ema, net, decay):
    for pe, p in zip(ema.parameters(), net.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for be, b in zip(ema.buffers(), net.buffers()):
        be.copy_(b)


def save_checkpoint(path, ema, meta):
    sd = {k: v.half() if v.is_floating_point() else v for k, v in ema.state_dict().items()}
    tmp = path + ".tmp"
    torch.save({"format": "diffusion-1", "state_dict": sd, "meta": meta}, tmp)
    os.replace(tmp, path)


def load_diffusion(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    net = DiffusionUNet(ch=meta["ch"], mult=tuple(meta["mult"]), n_res=meta["n_res"], dropout=0.0)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return net.to(device).eval(), meta


# =========================================================================== metrics
def lapvar(rgb01):
    """Variance of the Laplacian of the gray image (as _sharpness.py; higher = sharper)."""
    g = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def image_metrics(pred_rgb, gt_rgb):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    return (peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=1.0),
            structural_similarity(gt_rgb, pred_rgb, data_range=1.0, channel_axis=2))


class LPIPS:
    def __init__(self, device):
        self.fn = None
        try:
            import lpips
            self.fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:  # noqa: BLE001 - optional metric
            print(f"[eval] LPIPS unavailable ({type(e).__name__}: {e}); skipping it")

    @torch.no_grad()
    def __call__(self, pred, gt, device):
        if self.fn is None:
            return [float("nan")] * len(pred)
        p = torch.from_numpy(np.ascontiguousarray(pred)).permute(0, 3, 1, 2).float().to(device) * 2 - 1
        g = torch.from_numpy(np.ascontiguousarray(gt)).permute(0, 3, 1, 2).float().to(device) * 2 - 1
        return self.fn(p, g).flatten().cpu().tolist()


# =========================================================================== training
def wall_hours(t0):
    return (time.time() - t0) / 3600


@torch.no_grad()
def quick_val(sampler_net, diff, part, idx, dev, args):
    g, rgb = part.tensors(idx, dev)
    x = diff.sample(sampler_net, condition(g), args.val_ddim, args.amp, seed=1234)
    lab, gt = decode_lab(x), rgb_to_lab_t(rgb)
    ab = (lab[:, 1:] - gt[:, 1:]).pow(2).mean().sqrt().item()
    L = (lab[:, :1] - gt[:, :1]).pow(2).mean().sqrt().item()
    return ab, L


def train(args, data, dev, out):
    net = DiffusionUNet(ch=args.ch, mult=tuple(args.mult), n_res=args.n_res, dropout=args.dropout).to(dev)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    multi = dev.type == "cuda" and torch.cuda.device_count() > 1
    run_net = nn.DataParallel(net) if multi else net
    run_ema = nn.DataParallel(ema) if multi else ema
    diff = Diffusion(device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    meta = {"ch": args.ch, "mult": list(args.mult), "n_res": args.n_res, "ab_scale": AB_SCALE,
            "condition": "sar3: raw, gauss sigma 1, gauss sigma 2.5 (x2-1)", "schedule": "cosine",
            "T": diff.T, "params": n_params(net)}
    print(f"[train] UNet {n_params(net) / 1e6:.1f}M params | GPUs: "
          f"{torch.cuda.device_count() if dev.type == 'cuda' else 0} | budget {args.hours} h", flush=True)

    tr, rng = data["train"], np.random.RandomState(args.seed)
    val_idx = np.sort(rng.choice(len(data["val"]), min(args.n_val_quick, len(data["val"])), replace=False))
    hist = {"step": [], "loss": [], "lr": [], "hours": [], "val_step": [], "val_ab_rmse": [],
            "val_L_rmse": [], "val_hours": []}
    t0, step, last_val, run_loss = time.time(), 0, time.time(), []
    budget = args.hours * 3600

    def phase():
        full = (time.time() - t0) >= args.full_frac * budget
        return (256, args.bs_full) if full else (args.crop, args.bs)

    def train_step(idx, size):
        g, rgb = tr.tensors(idx, dev)
        c, x0 = condition(g), encode(rgb)
        c, x0 = augment(c, x0)
        c, x0 = random_crop([c, x0], size)
        t = torch.randint(0, diff.T, (x0.shape[0],), device=dev)
        noise = torch.randn_like(x0)
        with torch.autocast("cuda", enabled=args.amp):
            eps = run_net(torch.cat([diff.q_sample(x0, t, noise), c], 1), t)
        loss = F.mse_loss(eps.float(), noise)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return loss.detach()

    done = False
    while not done:
        size, bs = phase()
        perm = rng.permutation(len(tr))
        for s in range(0, len(perm) - bs + 1, bs):
            frac = (time.time() - t0) / budget
            if frac >= 1.0:
                done = True
                break
            if phase() != (size, bs):          # switch to full tiles: restart the epoch at the new batch size
                break
            warm = min(1.0, (step + 1) / args.warmup)
            lr = args.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
            for gp in opt.param_groups:
                gp["lr"] = lr
            net.train()
            try:
                loss = train_step(np.sort(perm[s:s + bs]), size)
            except torch.OutOfMemoryError:
                # out of GPU memory: halve this phase's batch size and carry on
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if size >= 256:
                    args.bs_full = max(2, args.bs_full // 2)
                else:
                    args.bs = max(4, args.bs // 2)
                print(f"[train] out of memory at {size}px x{bs} -> batch {args.bs_full if size >= 256 else args.bs}",
                      flush=True)
                break
            ema_update(ema, net, min(args.ema, (1 + step) / (10 + step)))
            step += 1
            run_loss.append(loss.item())
            if step % args.log_every == 0:
                hist["step"].append(step)
                hist["loss"].append(float(np.mean(run_loss)))
                hist["lr"].append(lr)
                hist["hours"].append(wall_hours(t0))
                print(f"step {step:7d}  loss {np.mean(run_loss):.4f}  lr {lr:.2e}  {size}px x{bs}  "
                      f"{wall_hours(t0):.2f} h", flush=True)
                run_loss = []
            if time.time() - last_val >= args.val_every_min * 60:
                ab, L = quick_val(run_ema, diff, data["val"], val_idx, dev, args)
                hist["val_step"].append(step)
                hist["val_ab_rmse"].append(ab)
                hist["val_L_rmse"].append(L)
                hist["val_hours"].append(wall_hours(t0))
                print(f"[val] step {step}  a/b RMSE {ab:.2f}  L RMSE {L:.2f}  ({len(val_idx)} val tiles, "
                      f"{args.val_ddim} DDIM steps)", flush=True)
                meta.update(steps=step, hours=wall_hours(t0))
                save_checkpoint(os.path.join(out, "model_diffusion.pth"), ema, meta)
                json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=1)
                last_val = time.time()
    meta.update(steps=step, hours=wall_hours(t0))
    save_checkpoint(os.path.join(out, "model_diffusion.pth"), ema, meta)
    json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=1)
    print(f"[train] done: {step} steps in {wall_hours(t0):.2f} h", flush=True)
    return ema, run_ema, diff, meta, hist


# =========================================================================== evaluation
ROWS = ("sample", "mean_of_K", "no_color", "mean_color")


@torch.no_grad()
def evaluate(args, run_ema, diff, data, dev, out, mean_ab, deadline):
    te = data["test"]
    lp = LPIPS(dev)
    n = len(te)
    recs = {r: [] for r in ROWS}
    first_samples, gt_lab_all = [], []
    t_eval = time.time()

    def score(row, lab, gt_rgb, gt_lab):
        rgb = lab_to_rgb_np(lab.cpu())
        ab = (lab[:, 1:] - gt_lab[:, 1:]).pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy()
        L = (lab[:, :1] - gt_lab[:, :1]).pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy()
        lps = lp(rgb, gt_rgb, dev)
        for i in range(len(rgb)):
            p, s = image_metrics(rgb[i], gt_rgb[i])
            recs[row].append({"psnr": p, "ssim": s, "ab_rmse": float(ab[i]), "L_rmse": float(L[i]),
                              "sharpness": lapvar(rgb[i]), "lpips": lps[i]})
        return rgb

    # pass 1: one sample per tile (+ baselines built on its L)
    figs = {}
    for b0 in range(0, n, args.eval_bs):
        idx = np.arange(b0, min(n, b0 + args.eval_bs))
        g, rgb = te.tensors(idx, dev)
        gt_lab = rgb_to_lab_t(rgb)
        gt_rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()
        x = diff.sample(run_ema, condition(g), args.ddim_steps, args.amp, seed=args.seed + b0)
        lab = decode_lab(x)
        shown = score("sample", lab, gt_rgb, gt_lab)
        neutral = lab.clone()
        neutral[:, 1:] = 0
        score("no_color", neutral, gt_rgb, gt_lab)
        const = lab.clone()
        const[:, 1:] = mean_ab.view(1, 2, 1, 1)
        score("mean_color", const, gt_rgb, gt_lab)
        first_samples.append(x.half().cpu())
        gt_lab_all.append(idx)
        for i in idx:
            if te.terrain[i] not in figs or len(figs[te.terrain[i]]) < 2:
                figs.setdefault(te.terrain[i], []).append(i)
        if b0 == 0:
            per_batch = time.time() - t_eval
            print(f"[eval] pass 1: {per_batch:.0f} s for the first {len(idx)} tiles "
                  f"-> ~{per_batch * n / len(idx) / 60:.0f} min for {n}", flush=True)
    t1 = time.time() - t_eval
    print(f"[eval] pass 1 done in {t1 / 60:.1f} min", flush=True)

    # pass 2: K-1 more samples per tile, if they fit before the deadline
    K = args.n_samples
    left = deadline - time.time()
    if K > 1 and t1 * (K - 1) * 1.1 > left:
        K = max(1, 1 + int(left / (t1 * 1.1)))
        print(f"[eval] only time for K={K} samples per tile", flush=True)
    extra = {}
    if K > 1:
        for b0 in range(0, n, args.eval_bs):
            idx = np.arange(b0, min(n, b0 + args.eval_bs))
            g, rgb = te.tensors(idx, dev)
            gt_lab = rgb_to_lab_t(rgb)
            gt_rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()
            c = condition(g)
            xs = [first_samples[b0 // args.eval_bs].float().to(dev)]
            for k in range(1, K):
                xs.append(diff.sample(run_ema, c, args.ddim_steps, args.amp, seed=args.seed + 7919 * k + b0))
            mean_lab = decode_lab(torch.stack(xs).mean(0))
            m_rgb = score("mean_of_K", mean_lab, gt_rgb, gt_lab)
            for j, i in enumerate(idx):
                extra[int(i)] = m_rgb[j]
    else:
        recs.pop("mean_of_K")

    def agg(rows):
        keys = rows[0].keys()
        return {k: float(np.nanmean([r[k] for r in rows])) if rows else None for k in keys}

    gt_sharp = [lapvar(te.rgb[i] / 255.0) for i in range(n)]
    res = {"n": n, "eval_set": "test scenes " + str(sorted({scene_of(p[0]) for p in te.pairs}))
           + " (never trained on)", "resolution": 256, "ddim_steps": args.ddim_steps, "K": K,
           "ground_truth_sharpness": float(np.mean(gt_sharp))}
    for r, rows in recs.items():
        res[r] = agg(rows)
    terr = {}
    for t in sorted(set(te.terrain)):
        ii = [i for i in range(n) if te.terrain[i] == t]
        terr[t] = {"n": len(ii), **{r: agg([recs[r][i] for i in ii]) for r in recs},
                   "ground_truth_sharpness": float(np.mean([gt_sharp[i] for i in ii]))}
    res["per_terrain"] = terr
    json.dump(res, open(os.path.join(out, "metrics_test.json"), "w"), indent=2)

    print("\n" + "=" * 92)
    print(f"Conditional pixel diffusion - {n} test tiles from held-out scenes, 256 px, "
          f"{args.ddim_steps} DDIM steps, K={K}")
    print("=" * 92)
    print(f"  {'':22s}{'PSNR':>8s}{'SSIM':>8s}{'a/b RMSE':>10s}{'L RMSE':>9s}{'sharpness':>11s}{'LPIPS':>8s}")
    for r in recs:
        m = res[r]
        print(f"  {r:22s}{m['psnr']:8.2f}{m['ssim']:8.3f}{m['ab_rmse']:10.2f}{m['L_rmse']:9.2f}"
              f"{m['sharpness']:11.0f}{m['lpips']:8.3f}")
    print(f"  {'ground truth':22s}{'':8s}{'':8s}{'':10s}{'':9s}{res['ground_truth_sharpness']:11.0f}")
    print(f"\n  {'terrain':<12}{'n':>5}{'PSNR':>8}{'a/b RMSE':>10}{'mean-col':>10}{'L RMSE':>9}{'sharp':>8}{'GT sharp':>10}")
    for t, s in terr.items():
        print(f"  {t:<12}{s['n']:>5}{s['sample']['psnr']:>8.2f}{s['sample']['ab_rmse']:>10.2f}"
              f"{s['mean_color']['ab_rmse']:>10.2f}{s['sample']['L_rmse']:>9.2f}"
              f"{s['sample']['sharpness']:>8.0f}{s['ground_truth_sharpness']:>10.0f}")

    # figure: 2 tiles per terrain - SAR | sample | mean of K | ground truth
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    pick = [i for t in sorted(figs) for i in figs[t]]
    cols = 4 if extra else 3
    fig, axes = plt.subplots(len(pick), cols, figsize=(3.2 * cols, 3.2 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        x = first_samples[i // args.eval_bs][i % args.eval_bs:i % args.eval_bs + 1].float()
        panels = [te.sar[i], lab_to_rgb_np(decode_lab(x))[0]] + ([extra[i]] if extra else []) + [te.rgb[i]]
        titles = ["SAR input", "diffusion sample"] + ([f"mean of {K} samples"] if extra else []) + ["real optical"]
        for c, (im, tt) in enumerate(zip(panels, titles)):
            axes[r, c].imshow(im, cmap="gray" if im.ndim == 2 else None)
            axes[r, c].set_title(f"{tt}" + (f" ({te.terrain[i]})" if c == 0 else ""), fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sample_predictions.png"), dpi=110)
    plt.close(fig)
    return res


def plot_history(hist, out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["step"], hist["loss"])
    ax[0].set(title="training loss (eps MSE)", xlabel="step", yscale="log")
    ax[1].plot(hist["val_step"], hist["val_ab_rmse"], "o-", label="a/b RMSE")
    ax[1].plot(hist["val_step"], hist["val_L_rmse"], "o-", label="L RMSE")
    ax[1].set(title="validation (one sample, held-out scenes)", xlabel="step", ylabel="Lab units")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "loss_curve.png"), dpi=110)
    plt.close(fig)


# =========================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--split_file", default="scene_split.json")
    ap.add_argument("--out", default="diffusion_run")
    ap.add_argument("--hours", type=float, default=7.0, help="training wall-time budget")
    ap.add_argument("--session_hours", type=float, default=11.5,
                    help="hard stop for the whole script (Kaggle kills at 12 h); limits evaluation")
    ap.add_argument("--full_frac", type=float, default=0.75, help="train on 128 crops before this fraction")
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32, help="batch size on 128 crops")
    ap.add_argument("--bs_full", type=int, default=12, help="batch size on full 256 tiles")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--mult", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n_res", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n_train", type=int, default=0, help="0 = all training pairs")
    ap.add_argument("--n_test", type=int, default=0, help="0 = all test pairs")
    ap.add_argument("--n_val_quick", type=int, default=64)
    ap.add_argument("--val_every_min", type=float, default=30)
    ap.add_argument("--val_ddim", type=int, default=25)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=4, help="K draws per test tile (mean of K row)")
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_only", default=None, help="score an existing checkpoint, no training")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run of every code path")
    args = ap.parse_args()
    if args.smoke:
        args.n_train, args.n_test, args.n_val_quick = 48, 8, 4
        args.ch, args.mult, args.bs, args.bs_full, args.crop = 32, [1, 1, 2, 2], 4, 2, 64
        args.hours, args.full_frac, args.val_every_min = 0.02, 0.5, 0.4
        args.ddim_steps, args.val_ddim, args.n_samples, args.eval_bs, args.log_every = 4, 3, 2, 4, 5
        args.warmup = 10
    t_start = time.time()
    deadline = t_start + args.session_hours * 3600
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = dev.type == "cuda" and not args.no_amp
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    pairs = find_pairs(args.root)
    assert pairs, f"no s1/s2 pairs under {args.root}"
    parts = scene_split(pairs, args.split_file)
    parts["train"] = subsample(parts["train"], args.n_train, args.seed)
    parts["test"] = subsample(parts["test"], args.n_test, args.seed)
    parts["val"] = subsample(parts["val"], args.n_val_quick, args.seed)   # only used for progress checks
    json.dump({"rule": f"scene split ({args.split_file})",
               "scenes": {k: sorted({scene_of(p[0]) for p in v}) for k, v in parts.items()},
               "n": {k: len(v) for k, v in parts.items()},
               "train": sorted(f"{p[2]}/{os.path.basename(p[0])}" for p in parts["train"])},
              open(os.path.join(args.out, "split.json"), "w"), indent=1)
    train_scenes = {scene_of(p[0]) for p in parts["train"]}
    assert not train_scenes & {scene_of(p[0]) for p in parts["test"] + parts["val"]}, "scene leak"
    print("pairs: " + ", ".join(f"{k} {len(v)} ({len({scene_of(p[0]) for p in v})} scenes)"
                                for k, v in parts.items()), flush=True)

    t = time.time()
    data = {k: Part(v, args.workers) for k, v in parts.items() if k != "train" or not args.eval_only}
    print(f"loaded into memory in {time.time() - t:.0f} s", flush=True)

    if args.eval_only:
        ema, meta = load_diffusion(args.eval_only, dev)
        run_ema = nn.DataParallel(ema) if dev.type == "cuda" and torch.cuda.device_count() > 1 else ema
        diff, hist = Diffusion(device=dev), None
        mean_pool = Part(subsample(parts["train"], 500, 0), args.workers)
    else:
        ema, run_ema, diff, meta, hist = train(args, data, dev, args.out)
        plot_history(hist, args.out)
        mean_pool = data["train"]
    sel = np.random.RandomState(0).choice(len(mean_pool), min(500, len(mean_pool)), replace=False)
    _, rgb = mean_pool.tensors(np.sort(sel), dev)
    mean_ab = rgb_to_lab_t(rgb)[:, 1:].mean(dim=(0, 2, 3))
    print(f"mean training color: a={mean_ab[0]:.2f} b={mean_ab[1]:.2f}", flush=True)
    evaluate(args, run_ema, diff, data, dev, args.out, mean_ab, deadline)
    print(f"\ntotal {wall_hours(t_start):.2f} h -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()
