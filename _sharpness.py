"""
_sharpness.py - how sharp is Model 2's output, really?

Variance of the Laplacian (higher = sharper) for every image in the SAR
colorization pipeline, so "the result is not sharp enough" can be answered
with numbers instead of vibes. Also prototypes an unsharp-mask boost.

    python _sharpness.py
"""
import cv2
import numpy as np
import torch
from PIL import Image

import os

from colorize import CHECKPOINTS, colorize, load_models
from color_utils import find_pairs, unsharp

SIZE = 256  # normalize every image to this before measuring (fair comparison)


def lapvar(a):
    """Variance of the Laplacian - classic sharpness score (higher = sharper)."""
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    a = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(a, cv2.CV_64F).var())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc, dec, meta, dncnn = load_models("sar", device, checkpoint=CHECKPOINTS["sar2"])  # Model 2
m3 = (load_models("sar", device, checkpoint=CHECKPOINTS["sar3"])
      if os.path.exists(CHECKPOINTS["sar3"]) else None)
first_pair = {}
for sar, opt, terrain in find_pairs("v_2"):
    first_pair.setdefault(terrain, (sar, opt))

results = {}
for terrain in ("agri", "urban"):
    path, gt_path = first_pair[terrain]

    pred_dn, gray_dn, _ = colorize(path, "sar", enc, dec, meta, dncnn, device)   # default
    pred_ns, _, _ = colorize(path, "sar", enc, dec, meta, None, device)          # --no_denoise
    pred_m3 = colorize(path, "sar", *m3[:3], None, device)[0] if m3 else None
    raw_sar = np.array(Image.open(path).convert("L"))
    gt = np.array(Image.open(gt_path).convert("RGB"))

    results[terrain] = {
        "raw SAR (s1)": lapvar(raw_sar),
        "DnCNN-denoised L": lapvar(gray_dn),
        "output, no denoise": lapvar(pred_ns),
        "output (default)": lapvar(pred_dn),
        "output + unsharp": lapvar(unsharp(pred_dn)),
        **({"Model 3 output": lapvar(pred_m3)} if m3 else {}),
        "ground truth": lapvar(gt),
    }

    if terrain == "agri":
        panels = [pred_dn, pred_ns, unsharp(pred_dn)] + ([pred_m3] if m3 else []) + [gt]
        titles = (["Model 2 (DnCNN)", "Model 2, no denoise", "Model 2 + unsharp mask"]
                  + (["Model 3"] if m3 else []) + ["ground truth"])
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.6))
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
