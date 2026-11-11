"""
ColorSAR - Stage A: train the DnCNN front-end of Model 2

Reproduction of the first stage of the original gray_model.ipynb:
    - pairs        : SAR (s1) grayscale  ->  optical (s2) grayscale, native 256x256
    - architecture : DnCNN depth 17 / 64 channels, residual output x - net(x) (dncnn.py)
    - objective    : MSE, Adam lr=1e-3, 15 epochs
    - reported by the original authors: PSNR 16.24 dB, SSIM 0.2641

Note what this stage really is: not speckle denoising in the classical sense but a
SAR -> optical-brightness *translation* (the target is a different sensor's image).
The DnCNN architecture is kept for fidelity to the original project.

Splits: canonical hash split (splits.py) - the same test pairs Model 1/2 hold out.
The checkpoint with the best val loss is kept (not the last epoch).

Outputs (in --out, default model2_run/):
    dncnn.pth            best weights (used by Train_model2.py, evaluate.py, colorize.py)
    dncnn_split.json     train/val pairs used
    dncnn_history.json   per-epoch train/val loss
    dncnn_loss_curve.png
    dncnn_metrics.json   PSNR/SSIM on held-out pairs, plus the raw-SAR baseline

Usage:
    python train_dncnn.py --epochs 15
    python train_dncnn.py --n_images 64 --epochs 1 --out _smoke     # smoke test
    python train_dncnn.py --eval_only                               # just score dncnn.pth
"""
import argparse
import json
import os
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

from color_utils import find_pairs
from dncnn import DnCNN, load_dncnn
from splits import canonical_split, heldout, subsample, write_split

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


class GrayPairDataset(Dataset):
    """(SAR grayscale, optical grayscale) pairs as (1,H,W) float tensors in [0,1]."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _gray(path):
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx):
        sar_path, opt_path, _ = self.pairs[idx]
        return self._gray(sar_path), self._gray(opt_path)


def run_epoch(loader, model, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            loss = criterion(model(noisy), clean)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean PSNR/SSIM of DnCNN output and of the raw SAR input, vs the optical gray."""
    model.eval()
    acc = {"dncnn": [], "raw_sar": []}
    for noisy, clean in loader:
        den = model(noisy.to(device)).clamp(0.0, 1.0).cpu().numpy()
        noisy, clean = noisy.numpy(), clean.numpy()
        for i in range(clean.shape[0]):
            for k, x in (("dncnn", den[i, 0]), ("raw_sar", noisy[i, 0])):
                acc[k].append((peak_signal_noise_ratio(clean[i, 0], x, data_range=1.0),
                               structural_similarity(clean[i, 0], x, data_range=1.0)))
    return {k: {"psnr": float(np.mean([a for a, _ in v])), "ssim": float(np.mean([b for _, b in v]))}
            for k, v in acc.items()}


def write_metrics(model, pairs, split_path, out_dir, device, batch_size, num_workers, n_eval=0):
    test, desc = heldout(pairs, split_path)
    test = subsample(test, n_eval, seed=42)
    print(f"Scoring on {desc}")
    loader = DataLoader(GrayPairDataset(test), batch_size=batch_size, num_workers=num_workers)
    m = evaluate(model, loader, device)
    m.update(n_images=len(test), eval_set=desc,
             upstream_reported={"psnr": 16.24, "ssim": 0.2641})
    with open(os.path.join(out_dir, "dncnn_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print(f"DnCNN    : PSNR {m['dncnn']['psnr']:.4f} dB   SSIM {m['dncnn']['ssim']:.4f}")
    print(f"raw SAR  : PSNR {m['raw_sar']['psnr']:.4f} dB   SSIM {m['raw_sar']['ssim']:.4f}   (no network)")
    print("Original gray_model.ipynb reported:  PSNR 16.24 dB   SSIM 0.2641")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--n_images", type=int, default=10000,
                    help="training pairs (original: 10000; 0 = whole train split)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="model2_run")
    ap.add_argument("--eval_only", action="store_true",
                    help="score the existing <out>/dncnn.pth on held-out pairs and exit")
    ap.add_argument("--eval_images", type=int, default=0,
                    help="held-out pairs to score (0 = all of them)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    ckpt = os.path.join(args.out, "dncnn.pth")
    split_path = os.path.join(args.out, "dncnn_split.json")

    if args.eval_only:
        write_metrics(load_dncnn(ckpt, device), pairs, split_path, args.out, device,
                      args.batch_size, args.num_workers, args.eval_images)
        raise SystemExit

    split = canonical_split(pairs)
    train_pairs = subsample(split["train"], args.n_images, args.seed)
    val_pairs = split["val"]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}  (test {len(split['test'])} held out)")
    write_split(split_path, train_pairs, val_pairs, "canonical (splits.canonical_split)")

    train_loader = DataLoader(GrayPairDataset(train_pairs), batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(GrayPairDataset(val_pairs), batch_size=args.batch_size,
                            num_workers=args.num_workers)

    model = DnCNN().to(device)
    print(f"DnCNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val, t0 = float("inf"), time.time()
    for epoch in range(args.epochs):
        tr = run_epoch(train_loader, model, criterion, device, optimizer)
        va = run_epoch(val_loader, model, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        flag = ""
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), ckpt)
            flag = "  <- best, saved"
        print(f"Epoch {epoch+1}/{args.epochs}  train={tr:.5f}  val={va:.5f}  "
              f"({(time.time()-t0)/60:.1f} min){flag}")

    with open(os.path.join(args.out, "dncnn_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    ep = range(1, args.epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, history["train_loss"], marker="o", label="train")
    plt.plot(ep, history["val_loss"], marker="o", label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("DnCNN (SAR gray -> optical gray)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "dncnn_loss_curve.png"), dpi=130); plt.close()

    if history["val_loss"][-1] < min(history["val_loss"][:-1] or [float("inf")]):
        print("note: val loss was still improving at the last epoch - more epochs may help")
    write_metrics(load_dncnn(ckpt, device), pairs, split_path, args.out, device,
                  args.batch_size, args.num_workers, args.eval_images)
    print("Next:  python Train_model2.py   (uses this checkpoint as its input stage)")
