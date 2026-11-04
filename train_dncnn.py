"""
ColorSAR - Stage A: train the DnCNN denoiser (Model 2 front-end)

Reproduction of the denoising stage from the original gray_model.ipynb:
    - pairs        : SAR (s1) grayscale  ->  optical (s2) grayscale
    - architecture : DnCNN depth 17 / 64 channels, residual learning (dncnn.py)
    - objective    : MSE, Adam lr=1e-3, 15 epochs, 80/20 split
    - reported by the original authors: PSNR 16.24 dB, SSIM 0.2641

Outputs (in --out, default model2_run/):
    dncnn.pth     trained weights (used by Train_model2.py and evaluate.py)
    history.json  per-epoch train/val loss
    loss_curve.png

Usage (laptop with the RTX 4050):
    python train_dncnn.py --epochs 15
Quick smoke test:
    python train_dncnn.py --n_images 64 --epochs 1
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
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset, random_split

from color_utils import find_pairs
from dncnn import DnCNN

matplotlib.use("Agg")


class GrayPairDataset(Dataset):
    """(SAR grayscale, optical grayscale) pairs as (1,H,W) float tensors in [0,1]."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _gray(self, path):
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx):
        sar_path, opt_path, _ = self.pairs[idx]
        return self._gray(sar_path), self._gray(opt_path)


def run_epoch(loader, model, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total = 0.0
    with torch.set_grad_enabled(training):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            loss = criterion(out, clean)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Average PSNR / SSIM of the denoised SAR gray vs the optical gray."""
    model.eval()
    psnr_sum = ssim_sum = n = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        den = model(noisy).clamp(0.0, 1.0).cpu().numpy()
        clean = clean.cpu().numpy()
        for i in range(clean.shape[0]):
            psnr_sum += peak_signal_noise_ratio(clean[i, 0], den[i, 0], data_range=1.0)
            ssim_sum += structural_similarity(clean[i, 0], den[i, 0], data_range=1.0)
            n += 1
    return psnr_sum / max(n, 1), ssim_sum / max(n, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--n_images", type=int, default=10000, help="pairs used (original: 10000)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="model2_run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    print(f"Found {len(pairs)} SAR/optical pairs in {args.root}")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(pairs))          # seeded shuffle => covers all 4 terrains
    pairs = [pairs[i] for i in perm[:args.n_images]]
    print(f"Using {len(pairs)} pairs")

    dataset = GrayPairDataset(pairs)
    n_test = max(int(0.2 * len(dataset)), 1)
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=args.num_workers)
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

    model = DnCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DnCNN parameters: {n_params:,}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    for epoch in range(args.epochs):
        tr = run_epoch(train_loader, model, criterion, device, optimizer)
        va = run_epoch(test_loader, model, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={tr:.4f}  val_loss={va:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)")

    torch.save(model.state_dict(), os.path.join(args.out, "dncnn.pth"))
    json.dump(history, open(os.path.join(args.out, "dncnn_history.json"), "w"), indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), history["train_loss"], marker="o", label="Train")
    plt.plot(range(1, args.epochs + 1), history["val_loss"], marker="o", label="Test")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("DnCNN denoising")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "dncnn_loss_curve.png"), dpi=130); plt.close()

    psnr, ssim = evaluate(model, test_loader, device)
    print(f"\nDenoiser on test split:  PSNR {psnr:.4f} dB   SSIM {ssim:.4f}")
    print("Original gray_model.ipynb reported:  PSNR 16.24 dB   SSIM 0.2641")
    print(f"\nSaved: {os.path.join(args.out, 'dncnn.pth')}, dncnn_history.json, dncnn_loss_curve.png")
    print("Next:  python Train_model2.py   (uses this checkpoint as its input stage)")
