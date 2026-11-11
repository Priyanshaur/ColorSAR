"""
_sharpness.py - how sharp is Model 2's output, really?

Variance of the Laplacian (higher = sharper) for every image in the SAR
colorization pipeline, so "the result is not sharp enough" can be answered
with numbers instead of vibes. Also prototypes an unsharp-mask boost.

    python _sharpness.py
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image

from colorize import colorize
from color_utils import load_decoder, unsharp
from dncnn import load_dncnn
from model_architecture import EnsembleEncoder

SIZE = 256  # normalize every image to this before measuring (fair comparison)


def lapvar(a):
    """Variance of the Laplacian - classic sharpness score (higher = sharper)."""
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    a = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(a, cv2.CV_64F).var())


def pair_of(s1_path):
    d, f = os.path.split(s1_path)
    return os.path.join(os.path.dirname(d), "s2", f.replace("_s1_", "_s2_"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = EnsembleEncoder(pretrained=True).to(device).eval()
for p in enc.parameters():
    p.requires_grad = False
dec = load_decoder("model2_run/model_2.pth", device)
dncnn = load_dncnn("model2_run/dncnn.pth", device)

results = {}
for terrain in ("agri", "urban"):
    s1 = sorted(os.listdir(os.path.join("v_2", terrain, "s1")))[0]
    path = os.path.join("v_2", terrain, "s1", s1)
    gt_path = pair_of(path)

    pred_dn, gray_dn, _ = colorize(path, "sar", enc, dec, dncnn, device)      # default
    pred_ns, _, _ = colorize(path, "sar", enc, dec, None, device)             # --no_denoise
    raw_sar = np.array(Image.open(path).convert("L"))
    gt = np.array(Image.open(gt_path).convert("RGB"))

    results[terrain] = {
        "raw SAR (s1)": lapvar(raw_sar),
        "DnCNN-denoised L": lapvar(gray_dn),
        "output, no denoise": lapvar(pred_ns),
        "output (default)": lapvar(pred_dn),
        "output + unsharp": lapvar(unsharp(pred_dn)),
        "ground truth": lapvar(gt),
    }

    if terrain == "agri":
        panels = [pred_dn, pred_ns, unsharp(pred_dn), gt]
        titles = ["default (DnCNN)", "no denoise", "+ unsharp mask", "ground truth"]
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
        for ax, im, t in zip(axes, panels, titles):
            ax.imshow(im)
            ax.set_title(f"{t}\nsharpness {lapvar(im):.0f}")
            ax.axis("off")
        fig.tight_layout()
        plt.savefig("demo_sharp_agri.png", dpi=130, bbox_inches="tight")
        plt.close()
        print("[OK] figure -> demo_sharp_agri.png")

print()
gt_scores = {t: r["ground truth"] for t, r in results.items()}
print(f"{'image':24s} {'agri':>18s} {'urban':>18s}")
print("-" * 62)
for key in results["agri"]:
    row = "  ".join(f"{results[t][key]:8.0f} ({100 * results[t][key] / gt_scores[t]:3.0f}% GT)"
                    for t in ("agri", "urban"))
    print(f"{key:24s} {row}")
