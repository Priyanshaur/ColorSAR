"""
ColorSAR - Stage B: train Model 2 (colorization from SAR input)

Reproduction of the colorization stage from the original gray_model.ipynb:
    - input  : luminance L of the *denoised* SAR (s1) image  (DnCNN from Stage A)
    - target : a/b channels of the paired optical (s2) image
    - model  : EnsembleEncoder (ResNet50 + DenseNet121, ImageNet weights, FROZEN)
               + U-Net Decoder (trained), exactly as Model 1
    - setup  : MSE, Adam lr=1e-3, 15 epochs, batch 32   (as in gray_model.ipynb)
    - reported by the original authors: PSNR 14.4466, SSIM 0.2314

Outputs (in --out, default model2_run/):
    model_2.pth            best decoder weights
    history.json           per-epoch train/val loss
    loss_curve.png         loss progression
    sample_predictions.png input gray vs ground truth vs prediction

Usage (laptop with the RTX 4050):
    python train_dncnn.py            # once, writes model2_run/dncnn.pth
    python Train_model2.py
Without the denoising stage (ablation, raw SAR luminance):
    python Train_model2.py --no_denoise
Quick smoke test:
    python Train_model2.py --n_images 32 --epochs 1 --no_denoise
"""
import argparse
import json
import os
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from color_utils import find_pairs, gray_to_L, lab_to_rgb, rgb_to_lab
from dncnn import load_dncnn
from model_architecture import Decoder, EnsembleEncoder

matplotlib.use("Agg")


class SARColorizationDataset(Dataset):
    """SAR (s1) grayscale in [0,1] + optical (s2) ab targets, both resized to 224."""

    def __init__(self, pairs, size=224):
        self.pairs = pairs
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sar_path, opt_path, _ = self.pairs[idx]

        gray = np.array(Image.open(sar_path).convert("L"), dtype=np.float32) / 255.0
        gray = cv2.resize(gray, (self.size, self.size))
        gray_t = torch.from_numpy(gray).unsqueeze(0)          # (1,H,W) in [0,1]

        rgb = np.array(Image.open(opt_path).convert("RGB"))
        rgb = cv2.resize(rgb, (self.size, self.size))
        _, ab = rgb_to_lab(rgb)                               # (2,H,W) in [-1,1]
        rgb_t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) / 255.0

        return gray_t, torch.from_numpy(ab), rgb_t


def run_epoch(loader, encoder, decoder, dncnn, criterion, device, optimizer=None):
    training = optimizer is not None
    decoder.train() if training else decoder.eval()
    encoder.eval()
    total = 0.0
    with torch.set_grad_enabled(training):
        for gray, ab, _ in loader:
            gray, ab = gray.to(device), ab.to(device)
            if dncnn is not None:
                with torch.no_grad():
                    gray = dncnn(gray).clamp(0.0, 1.0)        # speckle removed on GPU
            L = gray_to_L(gray)                              # (N,1,H,W) in [-1,1]
            inp = L.repeat(1, 3, 1, 1)
            with torch.no_grad():
                f56, f28, f14, f7 = encoder(inp)
            out = decoder(f7, f14, f28, f56)
            loss = criterion(out, ab)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def save_samples(loader, encoder, decoder, dncnn, device, path, n=4):
    encoder.eval(); decoder.eval()
    gray, ab, rgb_gt = next(iter(loader))
    gray, ab = gray[:n].to(device), ab[:n].to(device)
    if dncnn is not None:
        gray = dncnn(gray).clamp(0.0, 1.0)
    L = gray_to_L(gray)
    f56, f28, f14, f7 = encoder(L.repeat(1, 3, 1, 1))
    pred = decoder(f7, f14, f28, f56)

    pred_rgb = lab_to_rgb(L, pred)
    gt_rgb = rgb_gt[:n].permute(0, 2, 3, 1).numpy()
    in_gray = gray.cpu().numpy()[:, 0]

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i in range(n):
        axes[0, i].imshow(in_gray[i], cmap="gray"); axes[0, i].set_title("Input (denoised SAR L)")
        axes[1, i].imshow(gt_rgb[i]);              axes[1, i].set_title("Ground truth (s2)")
        axes[2, i].imshow(pred_rgb[i]);            axes[2, i].set_title("Model 2 output")
        for r in range(3):
            axes[r, i].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--n_images", type=int, default=8000, help="pairs used (original: 8000)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dncnn", default="model2_run/dncnn.pth")
    ap.add_argument("--no_denoise", action="store_true",
                    help="use raw SAR luminance instead of the DnCNN output")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="model2_run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dncnn = None
    if not args.no_denoise:
        dncnn = load_dncnn(args.dncnn, device)
        print(f"Loaded DnCNN from {args.dncnn} (input will be denoised)")
    else:
        print("!! --no_denoise: training on RAW SAR luminance (weaker, for comparison)")

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    print(f"Found {len(pairs)} SAR/optical pairs in {args.root}")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm[:args.n_images]]
    print(f"Using {len(pairs)} pairs")

    dataset = SARColorizationDataset(pairs)
    n_val = max(int(0.2 * len(dataset)), 1)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=args.num_workers)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    encoder = EnsembleEncoder(pretrained=True).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    decoder = Decoder().to(device)
    print(f"Trainable decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        tr = run_epoch(train_loader, encoder, decoder, dncnn, criterion, device, optimizer)
        va = run_epoch(val_loader, encoder, decoder, dncnn, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={tr:.4f}  val_loss={va:.4f}  "
              f"({(time.time()-t0)/60:.1f} min elapsed)")
        if va < best_val:
            best_val = va
            torch.save(decoder.state_dict(), os.path.join(args.out, "model_2.pth"))

    json.dump(history, open(os.path.join(args.out, "history.json"), "w"), indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), history["train_loss"], marker="o", label="Train")
    plt.plot(range(1, args.epochs + 1), history["val_loss"], marker="o", label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("Model 2 training")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "loss_curve.png"), dpi=130); plt.close()

    decoder.load_state_dict(torch.load(os.path.join(args.out, "model_2.pth"),
                                       map_location=device))
    save_samples(val_loader, encoder, decoder, dncnn, device,
                 os.path.join(args.out, "sample_predictions.png"))
    print(f"\nDone. Results in ./{args.out}/")
    print("Metrics next:  python evaluate.py --model 2 --checkpoint model2_run/model_2.pth")
