"""
ColorSAR - colorizer training (Models 1, 2 and 3 share this one script)

    Model 1 : L of the optical (s2) image  -> a/b of the same image   (clean input)
    Model 2 : L of the DnCNN-processed SAR (s1) image -> a/b of the paired s2 image
    Model 3 : the SAR (s1) tile itself (3 despeckling levels) -> L AND a/b of the s2 image

All: EnsembleEncoder (ResNet50 + DenseNet121, ImageNet weights, frozen backbones) +
U-Net decoder. Train_model1.py / Train_model2.py / Train_model3.py call this with each
model's defaults; the Kaggle notebook/kernel runs it too.

Splits: the canonical hash split from splits.py (80/10/10). The test pairs are never
touched here; --n_images / --n_val only subsample the train / val portions.

Outputs (in --out):
    model_<k>.pth           best checkpoint (format 2: decoder + encoder adapters + meta)
    split.json              exactly which pairs were used for train / val
    history.json            per-epoch losses, val color error (a/b RMSE) and learning rate
    loss_curve.png          loss / val error progression
    sample_predictions.png  input / ground truth / prediction on val images

The best epoch is chosen by validation a/b RMSE in Lab units (plus L RMSE when L is
predicted) - a metric that means the same thing whatever the training loss is.

Options (all recorded in the checkpoint's meta, so inference applies them automatically):
    --loss mse|l1|bins  a/b objective; "bins" = class-rebalanced color-bin classification
    --predict_L         also predict the L channel (Model 3 default)
    --train_adapters    also train the encoder's 1x1 projections + fusion blocks
    --imagenet_norm     ImageNet-normalized backbone input (always on for Model 3)
    --augment           random 90-degree rotations + flips (satellite tiles have no "up")
"""
import argparse
import json
import os
import time

import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from color_utils import PairDataset, ab_to_lab_units, find_pairs, lab_to_rgb, rgb_to_L
from dncnn import load_dncnn
from model_architecture import (Decoder, EnsembleEncoder, adapter_parameters,
                                decoder_out_channels, encoder_fingerprint, encoder_mode,
                                freeze_encoder, save_checkpoint)
from pipeline import (INPUT_OF_MODEL, bin_loss, class_weights, compute_bins, decode_output,
                      forward_raw, model_input)
from splits import canonical_split, subsample, write_split

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DEFAULTS = {
    # Model 1: the Kaggle 2x T4 run that produced model1_run/model_1.pth
    1: dict(n_images=0, epochs=20, batch_size=64, lr=5e-4, scheduler="cosine", out="model1_run",
            loss="mse", predict_L=False, train_adapters=False, imagenet_norm=False, augment=False),
    # Model 2: gray_model.ipynb recipe, the laptop run that produced model2_run/model_2.pth
    2: dict(n_images=8000, epochs=15, batch_size=32, lr=1e-3, scheduler="none", out="model2_run",
            loss="mse", predict_L=False, train_adapters=False, imagenet_norm=False, augment=False),
    # Model 3: direct SAR -> optical Lab, with every improvement on
    3: dict(n_images=0, epochs=30, batch_size=64, lr=5e-4, scheduler="cosine", out="model3_run",
            loss="bins", predict_L=True, train_adapters=True, imagenet_norm=True, augment=True),
}


def build_parser(model):
    d = DEFAULTS[model]
    ap = argparse.ArgumentParser(description=f"Train ColorSAR Model {model}")
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--n_images", type=int, default=d["n_images"],
                    help="training pairs to use (0 = the whole canonical train split)")
    ap.add_argument("--n_val", type=int, default=0,
                    help="validation pairs to use (0 = the whole canonical val split)")
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--batch_size", type=int, default=d["batch_size"],
                    help="total batch (split across GPUs with DataParallel)")
    ap.add_argument("--lr", type=float, default=d["lr"])
    ap.add_argument("--scheduler", choices=["none", "cosine"], default=d["scheduler"])
    ap.add_argument("--loss", choices=["mse", "l1", "bins"], default=d["loss"])
    ap.add_argument("--bin_grid", type=float, default=10.0, help="color bin size in Lab units")
    ap.add_argument("--T", type=float, default=0.38, help="annealed-mean temperature (bins)")
    ap.add_argument("--L_weight", type=float, default=1.0, help="weight of the L1 loss on L")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=d["out"])
    for flag in ("predict_L", "train_adapters", "imagenet_norm", "augment"):
        ap.add_argument(f"--{flag}", dest=flag, action="store_true")
        ap.add_argument(f"--no_{flag}", dest=flag, action="store_false")
        ap.set_defaults(**{flag: d[flag]})
    if model == 2:
        ap.add_argument("--dncnn", default="model2_run/dncnn.pth")
        ap.add_argument("--no_denoise", action="store_true",
                        help="ablation: raw SAR luminance instead of the DnCNN output")
    return ap


class Colorizer(nn.Module):
    """encoder + decoder in one module so DataParallel can split the batch."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, enc_in):
        return forward_raw(self.encoder, self.decoder, enc_in)


def augment(gray, rgb, ab):
    """Same random rot90 + flip for the whole batch (cheap, on device)."""
    k = int(torch.randint(4, (1,)))
    flip = bool(torch.randint(2, (1,)))
    out = []
    for x in (gray, rgb, ab):
        x = torch.rot90(x, k, dims=(2, 3))
        out.append(torch.flip(x, dims=(3,)) if flip else x)
    return out


class Objective:
    """Training loss + a loss-independent validation error, for one model config."""

    def __init__(self, meta, centers=None, weights=None, L_weight=1.0):
        self.meta, self.centers, self.weights, self.L_weight = meta, centers, weights, L_weight

    def loss(self, raw, ab, L_true):
        m = self.meta
        if m["predict_L"]:
            loss_L = (torch.tanh(raw[:, :1]) - L_true).abs().mean()
            rest = raw[:, 1:]
        else:
            loss_L, rest = 0.0, raw
        if m["ab_head"] == "bins":
            loss_ab = bin_loss(rest, ab, self.centers, self.weights)
        else:
            pred = rest if m["ab_head"] == "tanh" else torch.tanh(rest)
            loss_ab = ((pred - ab) ** 2).mean() if m["loss"] == "mse" else (pred - ab).abs().mean()
        return loss_ab + self.L_weight * loss_L

    @torch.no_grad()
    def errors(self, raw, ab, L_true, chunk=8):
        """Per-batch sums of squared a/b and L errors in Lab units (decoded output)."""
        se_ab = se_L = 0.0
        for i in range(0, raw.shape[0], chunk):
            L_p, ab_p = decode_output(raw[i:i + chunk], self.meta, self.centers)
            d = ab_to_lab_units(ab_p) - ab_to_lab_units(ab[i:i + chunk])
            se_ab += d.pow(2).mean(dim=(1, 2, 3)).sum().item()
            if L_p is not None:
                se_L += ((L_p - L_true[i:i + chunk]) * 50).pow(2).mean(dim=(1, 2, 3)).sum().item()
        return se_ab, se_L


def run_epoch(loader, net, parallel, meta, obj, dncnn, device, optimizer=None, aug=False):
    training = optimizer is not None
    net.decoder.train(training)
    encoder_mode(net.encoder, training, meta["train_adapters"])
    tot, n_batches, se_ab, se_L, n = 0.0, 0, 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for gray, rgb, ab, _ in loader:
            gray, rgb, ab = (t.to(device, non_blocking=True) for t in (gray, rgb, ab))
            if training and aug:
                gray, rgb, ab = augment(gray, rgb, ab)
            enc_in, _ = model_input(meta, gray=gray, rgb=rgb, dncnn=dncnn)
            L_true = rgb_to_L(rgb) if meta["predict_L"] else None
            raw = parallel(enc_in)
            loss = obj.loss(raw, ab, L_true)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                a, b = obj.errors(raw, ab, L_true)
                se_ab += a; se_L += b; n += raw.shape[0]
            tot += loss.item(); n_batches += 1
    loss = tot / max(n_batches, 1)
    if training:
        return loss, None, None
    return loss, (se_ab / max(n, 1)) ** 0.5, (se_L / max(n, 1)) ** 0.5


@torch.no_grad()
def save_samples(loader, net, meta, dncnn, device, path, n=4):
    net.eval()
    gray, rgb, ab, _ = next(iter(loader))
    gray, rgb = gray[:n].to(device), rgb[:n].to(device)
    enc_in, L_given = model_input(meta, gray=gray, rgb=rgb, dncnn=dncnn)
    L_pred, ab_pred = decode_output(net(enc_in), meta)
    L = L_pred if L_pred is not None else L_given
    pred_rgb = lab_to_rgb(L, ab_pred)
    gt = rgb.permute(0, 2, 3, 1).cpu().numpy()
    if meta["input"] == "sar3":
        shown, label = gray[:, 0].cpu().numpy(), "SAR (s1)"
    else:
        shown = ((L_given[:, 0] + 1) / 2).cpu().numpy()
        label = {"optical_L": "optical L", "sar_dncnn_L": "denoised SAR L" if dncnn is not None
                 else "raw SAR L"}[meta["input"]]
    n = len(gt)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(shown[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Input ({label})")
        axes[1, i].imshow(gt[i]); axes[1, i].set_title("Ground truth (s2)")
        axes[2, i].imshow(pred_rgb[i]); axes[2, i].set_title(f"Model {meta['model']} output")
        for r in range(3):
            axes[r, i].axis("off")
    plt.suptitle(f"Model {meta['model']} - held-out samples (not used for training)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main(model, argv=None):
    args = build_parser(model).parse_args(argv)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}")
    print("Config:", {k: v for k, v in vars(args).items() if k != "root"})

    dncnn, dncnn_split = None, None
    if model == 2:
        if args.no_denoise:
            print("!! --no_denoise: training on RAW SAR luminance (ablation)")
        else:
            dncnn = load_dncnn(args.dncnn, device)
            print(f"Loaded DnCNN from {args.dncnn} (applied at native resolution)")
            cand = os.path.join(os.path.dirname(args.dncnn), "dncnn_split.json")
            if os.path.exists(cand):
                dncnn_split = os.path.relpath(cand, args.out)

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    split = canonical_split(pairs)
    train_pairs = subsample(split["train"], args.n_images, args.seed)
    val_pairs = subsample(split["val"], args.n_val, args.seed)
    print(f"{len(pairs)} pairs  ->  train {len(train_pairs)}  val {len(val_pairs)}  "
          f"(test {len(split['test'])} held out, never touched here)")
    write_split(os.path.join(args.out, "split.json"), train_pairs, val_pairs,
                "canonical (splits.canonical_split)", upstream=dncnn_split)

    input_kind = INPUT_OF_MODEL[model]
    load_sar = input_kind != "optical_L"
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(PairDataset(train_pairs, load_sar=load_sar), batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(PairDataset(val_pairs, load_sar=load_sar), batch_size=args.batch_size,
                            shuffle=False, **kw)

    # ---- output head / color bins
    ab_head = "bins" if args.loss == "bins" else ("reg" if args.predict_L else "tanh")
    centers = weights = None
    meta = {"model": model, "input": input_kind,
            "input_norm": "imagenet" if (args.imagenet_norm or input_kind == "sar3") else "none",
            "dncnn_res": "native", "denoised": dncnn is not None, "loss": args.loss,
            "ab_head": ab_head, "predict_L": args.predict_L, "train_adapters": args.train_adapters,
            "augment": args.augment, "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "scheduler": args.scheduler,
            "n_train": len(train_pairs)}
    if ab_head == "bins":
        sample = subsample(train_pairs, 1000, seed=1)
        ds = PairDataset(sample, load_sar=False)
        ab_s = torch.cat([ds[i][2][:, ::4, ::4].reshape(2, -1).T for i in range(len(ds))])
        centers, prior = compute_bins(ab_s, grid=args.bin_grid)
        weights = class_weights(centers, prior)
        centers, weights = centers.to(device), weights.to(device)
        meta.update(bins=centers.cpu().tolist(), T=args.T)
        print(f"Color bins: {len(centers)} occupied {args.bin_grid:g}-unit cells "
              f"(weights {weights.min():.2f}..{weights.max():.2f})")
    obj = Objective(meta, centers, weights, args.L_weight)

    encoder = EnsembleEncoder(pretrained=True).to(device)
    freeze_encoder(encoder, train_adapters=args.train_adapters)
    print(f"Encoder adapter fingerprint at init: {encoder_fingerprint(encoder)}"
          f"{'  (adapters will be trained)' if args.train_adapters else ''}")
    decoder = Decoder(decoder_out_channels(meta)).to(device)
    net = Colorizer(encoder, decoder)
    # kept outside `net` (assigning it as an attribute would register a module cycle)
    parallel = nn.DataParallel(net) if torch.cuda.device_count() > 1 else net

    params = list(decoder.parameters())
    if args.train_adapters:
        params += adapter_parameters(encoder)
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
                 if args.scheduler == "cosine" else None)

    ckpt = os.path.join(args.out, f"model_{model}.pth")
    hist = {k: [] for k in ("train_loss", "val_loss", "val_ab_rmse", "val_L_rmse", "val_score", "lr")}
    best, t0 = float("inf"), time.time()
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tr, _, _ = run_epoch(train_loader, net, parallel, meta, obj, dncnn, device, optimizer,
                             aug=args.augment)
        va, ab_rmse, L_rmse = run_epoch(val_loader, net, parallel, meta, obj, dncnn, device)
        if scheduler:
            scheduler.step()
        score = ab_rmse + (L_rmse if args.predict_L else 0.0)
        for k, v in zip(hist, (tr, va, ab_rmse, L_rmse, score, lr)):
            hist[k].append(v)
        flag = ""
        if score < best:
            best = score
            save_checkpoint(ckpt, encoder, decoder, dict(meta, best_epoch=epoch + 1, best_val=score))
            flag = "  <- best, saved"
        print(f"Epoch {epoch+1}/{args.epochs}  train={tr:.4f}  val={va:.4f}  "
              f"val ab RMSE={ab_rmse:.2f}" + (f"  L RMSE={L_rmse:.2f}" if args.predict_L else "")
              + f"  lr={lr:.1e}  ({(time.time()-t0)/60:.1f} min){flag}", flush=True)

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    best_ep = hist["val_score"].index(best) + 1
    ep = range(1, args.epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, hist["train_loss"], marker="o", ms=3, label="train")
    ax[0].plot(ep, hist["val_loss"], marker="s", ms=3, label="val")
    ax[0].set_ylabel(f"training objective ({args.loss}{' + L1(L)' if args.predict_L else ''})")
    ax[1].plot(ep, hist["val_ab_rmse"], marker="s", ms=3, label="val a/b RMSE")
    if args.predict_L:
        ax[1].plot(ep, hist["val_L_rmse"], marker="^", ms=3, label="val L RMSE")
    ax[1].set_ylabel("Lab units (lower = better)")
    for a in ax:
        a.axvline(best_ep, ls=":", c="gray"); a.set_xlabel("epoch"); a.grid(alpha=0.3); a.legend()
    fig.suptitle(f"Model {model} training - best epoch {best_ep}")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "loss_curve.png"), dpi=130); plt.close(fig)

    # reload the best checkpoint (decoder + adapters) for the sample figure
    from model_architecture import load_colorizer
    enc_b, dec_b, meta_b = load_colorizer(ckpt, device, encoder=encoder)
    save_samples(val_loader, Colorizer(enc_b, dec_b), meta_b, dncnn, device,
                 os.path.join(args.out, "sample_predictions.png"))
    print(f"\nDone. Best epoch {best_ep} (val score {best:.3f}). Results in ./{args.out}/")
    print(f"Metrics next:  python evaluate.py --model {model} --checkpoint {ckpt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", type=int, choices=[1, 2, 3], required=True)
    known, rest = ap.parse_known_args()
    main(known.model, rest)
