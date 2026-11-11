"""
ColorSAR - Unified inference: the end-to-end demo

This is the single entry point of the final system. It picks the right model for
whatever you feed it:

    mode "sar"   (single-channel / grayscale SAR image, e.g. v_2/*/s1/*.png)
        s1 -> DnCNN denoise -> L channel -> Model 2 colorizer -> RGB
        checkpoint: model2_run/model_2.pth  (+ model2_run/dncnn.pth)

    mode "gray"  (an RGB or grayscale optical photograph, e.g. sample_input.jpeg)
        L of the input -> Model 1 colorizer -> RGB
        checkpoint: model1_run/model_1.pth

    mode "auto"  (default): 1-band image -> "sar", 3-band image -> "gray".

If the input happens to be a file inside v_2/, the matching pair (s1 <-> s2) is
looked up automatically and shown as a ground-truth panel.

Usage:
    python colorize.py --image v_2/agri/s1/<file>.png
    python colorize.py --image sample_input.jpeg --mode gray
    python colorize.py --image v_2/urban/s1/<file>.png --out demo.png
    python colorize.py --image v_2/agri/s1/<file>.png --no_denoise   # ablation
    python colorize.py --image v_2/agri/s1/<file>.png --sharp         # cosmetic sharpening

Outputs (both written next to each other):
    <out>                 side-by-side figure (input / colorized / ground truth)
    <out stem>_raw.png    the colorized image alone, at the input's own resolution
"""
import argparse
import os

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from color_utils import find_pairs, gray_to_L, lab_to_rgb, load_decoder, rgb_to_L, unsharp
from dncnn import load_dncnn
from model_architecture import EnsembleEncoder

matplotlib.use("Agg")

SIZE = 224  # the colorizers were trained on 224x224 patches


@torch.no_grad()
def colorize(path, mode, encoder, decoder, dncnn, device):
    """Return (prediction float RGB at original resolution, input gray, original PIL)."""
    pil = Image.open(path)
    orig_w, orig_h = pil.size

    if mode == "sar":
        gray = np.array(pil.convert("L"), dtype=np.float32) / 255.0
        t = torch.from_numpy(cv2.resize(gray, (SIZE, SIZE))).unsqueeze(0).unsqueeze(0)
        t = t.to(device)
        if dncnn is not None:
            t = dncnn(t).clamp(0.0, 1.0)          # strip speckle, same as training
        L = gray_to_L(t)                          # (1,1,H,W) in [-1,1]
        shown_gray = t[0, 0].cpu().numpy()
    else:
        rgb = np.array(pil.convert("RGB"), dtype=np.float32)
        t = torch.from_numpy(cv2.resize(rgb, (SIZE, SIZE))).permute(2, 0, 1)
        t = (t / 255.0).unsqueeze(0).to(device)
        L = rgb_to_L(t)                           # (1,1,H,W) in [-1,1]
        shown_gray = t[0].mean(dim=0).cpu().numpy()

    f56, f28, f14, f7 = encoder(L.repeat(1, 3, 1, 1))
    pred_ab = decoder(f7, f14, f28, f56)

    pred = lab_to_rgb(L, pred_ab)[0]              # (H,W,3) float in [0,1]
    pred = cv2.resize(pred, (orig_w, orig_h))     # back to the input's resolution
    return np.clip(pred, 0.0, 1.0), shown_gray, pil


def find_ground_truth(path, root):
    """If `path` is one of the s1/s2 files in the dataset, return its counterpart."""
    path = os.path.abspath(path)
    for sar, opt, _ in find_pairs(root):
        if os.path.abspath(sar) == path:
            return opt, "Ground truth (paired s2)"
        if os.path.abspath(opt) == path:
            return sar, "Paired input (s1)"
    return None, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image (SAR s1 or RGB/grayscale photo)")
    ap.add_argument("--mode", choices=["auto", "sar", "gray"], default="auto",
                    help="auto (default): 1-band -> sar, 3-band -> gray")
    ap.add_argument("--out", default="colorized_result.png", help="figure to write")
    ap.add_argument("--checkpoint", default=None,
                    help="defaults to model2_run/model_2.pth (sar) / model1_run/model_1.pth (gray)")
    ap.add_argument("--dncnn", default="model2_run/dncnn.pth")
    ap.add_argument("--no_denoise", action="store_true",
                    help="sar mode: feed raw SAR luminance, skipping the DnCNN")
    ap.add_argument("--sharp", action="store_true",
                    help="apply an unsharp mask to the output (cosmetic sharpening, "
                         "cannot invent detail the luminance does not have)")
    ap.add_argument("--root", default="v_2", help="dataset root used to find a ground truth pair")
    args = ap.parse_args()

    assert os.path.exists(args.image), f"Input image not found: {args.image}"

    mode = args.mode
    if mode == "auto":
        bands = len(Image.open(args.image).getbands())
        mode = "sar" if bands == 1 else "gray"
        print(f"Auto-detected mode: {mode} ({bands}-band input)")

    if args.checkpoint is None:
        args.checkpoint = "model2_run/model_2.pth" if mode == "sar" else "model1_run/model_1.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dncnn = None
    if mode == "sar":
        if args.no_denoise:
            print("!! --no_denoise: feeding RAW SAR luminance (ablation)")
        else:
            dncnn = load_dncnn(args.dncnn, device)
            print(f"DnCNN loaded from {args.dncnn}")
    print(f"Model: {'2 (SAR input)' if mode == 'sar' else '1 (clean input)'}  "
          f"|  checkpoint: {args.checkpoint}")

    encoder = EnsembleEncoder(pretrained=True).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    from model_architecture import encoder_fingerprint
    print(f"Encoder frozen-layer fingerprint: {encoder_fingerprint(encoder)} "
          "(must match the training run that produced the checkpoint)")
    decoder = load_decoder(args.checkpoint, device)

    pred, gray_in, pil = colorize(args.image, mode, encoder, decoder, dncnn, device)
    if args.sharp:
        pred = unsharp(pred)
        print("Unsharp mask applied (--sharp)")

    raw_path = os.path.splitext(args.out)[0] + "_raw.png"
    Image.fromarray((pred * 255).astype(np.uint8)).save(raw_path)

    gt_path, gt_title = find_ground_truth(args.image, args.root)

    n_panels = 3 if gt_path else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(gray_in, cmap="gray")
    axes[0].set_title(f"Input ({'denoised SAR L' if mode == 'sar' else 'L'})")
    axes[1].imshow(pred)
    axes[1].set_title(f"ColorSAR {'Model 2' if mode == 'sar' else 'Model 1'} output")
    if gt_path:
        axes[2].imshow(np.array(Image.open(gt_path).convert("RGB")))
        axes[2].set_title(gt_title)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"[OK] figure      -> {args.out}")
    print(f"[OK] colorized   -> {raw_path}  ({pil.size[0]}x{pil.size[1]})")
