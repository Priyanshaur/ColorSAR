"""
Usage (PowerShell, from the folder containing v_2):
    python train_model1.py --n_images 2000 --epochs 3
"""
import os, argparse, time, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visual_test import EnsembleEncoder, Decoder  # same architecture as the original
from model_architecture import encoder_fingerprint


def find_optical_images(root_dir):
    """Collect optical (s2) PNG paths from every terrain folder."""
    paths = []
    for terrain in sorted(os.listdir(root_dir)):
        s2_dir = os.path.join(root_dir, terrain, "s2")
        if os.path.isdir(s2_dir):
            for f in sorted(os.listdir(s2_dir)):
                if f.endswith(".png"):
                    paths.append(os.path.join(s2_dir, f))
    return paths


def rgb_to_lab(img):
    lab = rgb2lab(img).astype("float32")
    lab = np.transpose(lab, (2, 0, 1))
    L = 2 * lab[[0], ...] / 100 - 1
    ab = 2 * (lab[[1, 2], ...] + 128) / 255 - 1
    return L, ab


class ColorizationDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        img = cv2.resize(img, (224, 224))
        L, ab = rgb_to_lab(img)
        return torch.from_numpy(L), torch.from_numpy(ab)


def run_epoch(loader, encoder, decoder, criterion, device, optimizer=None):
    training = optimizer is not None
    decoder.train() if training else decoder.eval()
    total = 0.0
    with torch.set_grad_enabled(training):
        for L, ab in loader:
            L, ab = L.to(device), ab.to(device)
            L = L.repeat(1, 3, 1, 1)
            with torch.no_grad():
                f56, f28, f14, f7 = encoder(L)
            out = decoder(f7, f14, f28, f56)
            loss = criterion(out, ab)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / max(len(loader), 1)


def save_samples(loader, encoder, decoder, device, path, n=4):
    encoder.eval(); decoder.eval()
    L, ab = next(iter(loader))
    L, ab = L[:n].to(device), ab[:n].to(device)
    with torch.no_grad():
        f56, f28, f14, f7 = encoder(L.repeat(1, 3, 1, 1))
        pred = decoder(f7, f14, f28, f56)

    def to_rgb(L_t, ab_t):
        L_u = (L_t.cpu().numpy() + 1) * 0.5 * 100
        ab_u = (ab_t.cpu().numpy() + 1) * 0.5 * 255 - 128
        lab = np.transpose(np.concatenate([L_u, ab_u], axis=0), (1, 2, 0))
        return lab2rgb(lab)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i in range(n):
        axes[0, i].imshow(L[i, 0].cpu().numpy(), cmap="gray"); axes[0, i].set_title("Input (L)")
        axes[1, i].imshow(to_rgb(L[i], ab[i])); axes[1, i].set_title("Real color")
        axes[2, i].imshow(to_rgb(L[i], pred[i])); axes[2, i].set_title("Predicted color")
        for r in range(3):
            axes[r, i].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--n_images", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="model1_run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = find_optical_images(args.root)
    print(f"Found {len(paths)} optical images in total")

    # random subset (fixed seed) so it covers all four terrains, not just the first folder
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(paths))[:args.n_images]
    paths = [paths[i] for i in idx]

    dataset = ColorizationDataset(paths)
    n_val = max(int(0.1 * len(dataset)), args.batch_size)  # ensure at least one full val batch
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=0)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    encoder = EnsembleEncoder(pretrained=True).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Encoder frozen-layer fingerprint: {encoder_fingerprint(encoder)} "
          "(evaluation must print the same value)")
    decoder = Decoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        tr = run_epoch(train_loader, encoder, decoder, criterion, device, optimizer)
        va = run_epoch(val_loader, encoder, decoder, criterion, device)
        history["train_loss"].append(tr); history["val_loss"].append(va)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={tr:.4f}  val_loss={va:.4f}  "
              f"({(time.time()-t0)/60:.1f} min elapsed)")
        if va < best_val:
            best_val = va
            torch.save(decoder.state_dict(), os.path.join(args.out, "model_1.pth"))

    json.dump(history, open(os.path.join(args.out, "history.json"), "w"), indent=2)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, args.epochs + 1), history["train_loss"], marker="o", label="Train")
    plt.plot(range(1, args.epochs + 1), history["val_loss"], marker="o", label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("Model 1 training (short run)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "loss_curve.png"), dpi=130); plt.close()

    # reload the best decoder for the sample figure
    decoder.load_state_dict(torch.load(os.path.join(args.out, "model_1.pth"), map_location=device))
    save_samples(val_loader, encoder, decoder, device, os.path.join(args.out, "sample_predictions.png"))
    print(f"\nDone. Results saved in ./{args.out}/  (loss_curve.png, sample_predictions.png, history.json, model_1.pth)")