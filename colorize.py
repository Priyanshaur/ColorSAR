"""
ColorSAR - Unified inference: the end-to-end demo

This is the single entry point of the final system (the dashboard and the helper
scripts import it). It picks the right model for whatever you feed it:

    mode "sar"   Sentinel-1 SAR tile (e.g. v_2/*/s1/*.png)
        Model 3 (default when model3_run/model_3.pth exists):
            s1 -> encoder-decoder -> predicted L + a/b -> RGB
        Model 2 (--sar_model 2, the original pipeline):
            s1 -> DnCNN -> L channel -> Model 2 colorizer -> RGB
            checkpoint: model2_run/model_2.pth  (+ model2_run/dncnn.pth)

    mode "gray"  an optical image: RGB or grayscale (e.g. v_2/*/s2/*.png)
        L of the input -> Model 1 colorizer -> RGB
        checkpoint: model1_run/model_1.pth

    mode "auto"  (default): "sar" when the file is a Sentinel-1 tile by name
        (`_s1_` in the filename or an `s1/` folder), otherwise "gray". The band count
        is NOT used: a grayscale photo is 1-band too. For other SAR files pass --mode sar.

Both models were trained only on 256x256 Sentinel-2 satellite tiles: everyday photos
(sample_input.jpeg) are outside that domain and will get satellite-like colors.

Output resolution: color (a/b) is predicted at 224x224 and upsampled; the luminance is
taken at the input's FULL resolution (for SAR: the DnCNN applied to the full tile), so
the output keeps all the detail its luminance has.

If the input is a file inside v_2/, its pair (s1 <-> s2) is shown as a ground-truth panel.

Usage:
    python colorize.py --image v_2/agri/s1/<file>.png
    python colorize.py --image sample_input.jpeg
    python colorize.py --image v_2/urban/s1/<file>.png --out demo.png
    python colorize.py --image v_2/agri/s1/<file>.png --no_denoise   # ablation
    python colorize.py --image v_2/agri/s1/<file>.png --sharp         # cosmetic sharpening

Outputs:
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

from color_utils import (SIZE, find_pairs, gray_to_L, is_sentinel1_path, lab_to_rgb, rgb_to_L,
                         sar_to_L, unsharp)
from dncnn import load_dncnn
from model_architecture import load_colorizer
from pipeline import decode_output, forward_raw, model_input

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

CHECKPOINTS = {"sar2": "model2_run/model_2.pth", "sar3": "model3_run/model_3.pth",
               "gray": "model1_run/model_1.pth", "gray1b": "model1b_run/model_1.pth"}
DNCNN_CKPT = "model2_run/dncnn.pth"


def default_checkpoint(mode, sar_model="auto", gray_model="auto"):
    """Newest available model for a mode ('auto'), or the one asked for."""
    if mode == "sar":
        if sar_model == "auto":
            sar_model = "3" if os.path.exists(CHECKPOINTS["sar3"]) else "2"
        return CHECKPOINTS["sar" + str(sar_model)]
    if gray_model == "auto":
        gray_model = "1b" if os.path.exists(CHECKPOINTS["gray1b"]) else "1"
    return CHECKPOINTS["gray1b" if str(gray_model) == "1b" else "gray"]


def detect_mode(path):
    return "sar" if is_sentinel1_path(path) else "gray"


def load_models(mode, device, checkpoint=None, dncnn_path=DNCNN_CKPT, denoise=True):
    """-> (encoder, decoder, meta, dncnn or None) for the given mode."""
    path = checkpoint or default_checkpoint(mode)
    encoder, decoder, meta = load_colorizer(path, device)
    meta.setdefault("input", "sar_dncnn_L" if mode == "sar" else "optical_L")
    meta["checkpoint"] = path
    needs_dncnn = meta["input"] == "sar_dncnn_L" and denoise
    dncnn = load_dncnn(dncnn_path, device) if needs_dncnn else None
    return encoder, decoder, meta, dncnn


def model_name(meta):
    """Display name: Model 1 / 1b (improved, format 2) / 2 / 3."""
    base = {"optical_L": "Model 1", "sar_dncnn_L": "Model 2", "sar3": "Model 3"}[meta["input"]]
    return base + ("b" if meta["input"] == "optical_L" and meta.get("format", 2) == 2 else "")


@torch.no_grad()
def colorize(path, mode, encoder, decoder, meta, dncnn, device):
    """Return (prediction float RGB at the input's resolution, luminance shown as input, PIL)."""
    pil = Image.open(path)
    kind = meta["input"]
    if kind == "optical_L" and mode == "sar":
        raise ValueError("a Model 1 checkpoint cannot colorize in sar mode")
    if kind == "optical_L":
        rgb = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
        small = cv2.resize(rgb, (SIZE, SIZE))                     # same resize as training
        enc_in, _ = model_input(meta, rgb=torch.from_numpy(small).permute(2, 0, 1)[None].to(device))
        L_full = rgb_to_L(torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device))
        shown = ((L_full[0, 0] + 1) / 2).cpu().numpy()
    else:
        gray = np.array(pil.convert("L"), dtype=np.float32) / 255.0
        g = torch.from_numpy(gray)[None, None].to(device)
        enc_in, _ = model_input(meta, gray=g, dncnn=dncnn)
        if kind == "sar_dncnn_L":
            _, native = sar_to_L(g, dncnn, meta["dncnn_res"], need_native=True)
            L_full = gray_to_L(native)                            # full-res denoised luminance
            shown = native[0, 0].cpu().numpy()
        else:
            L_full, shown = None, gray                            # Model 3 predicts its own L

    L_pred, pred_ab = decode_output(forward_raw(encoder, decoder, enc_in), meta)
    if L_full is None:                                            # Model 3: upsample predicted L
        L_full = torch.nn.functional.interpolate(L_pred, size=(pil.size[1], pil.size[0]),
                                                 mode="bilinear", align_corners=False)
    pred = lab_to_rgb(L_full, pred_ab)[0]       # a/b upsampled to the full-res L
    return pred, shown, pil


def find_ground_truth(path, root):
    """If `path` is one of the s1/s2 files in the dataset, return its counterpart."""
    path = os.path.abspath(path)
    for sar, opt, _ in find_pairs(root):
        if os.path.abspath(sar) == path:
            return opt, "Ground truth (paired s2)"
        if os.path.abspath(opt) == path:
            return sar, "Paired input (s1)"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image (SAR s1 or RGB/grayscale photo)")
    ap.add_argument("--mode", choices=["auto", "sar", "gray"], default="auto",
                    help="auto (default): Sentinel-1 file name/folder -> sar, otherwise gray")
    ap.add_argument("--out", default="colorized_result.png", help="figure to write")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit checkpoint (overrides --sar_model / --gray_model)")
    ap.add_argument("--sar_model", choices=["auto", "2", "3"], default="auto",
                    help="sar mode: 3 = direct SAR model (default if trained), 2 = DnCNN pipeline")
    ap.add_argument("--gray_model", choices=["auto", "1", "1b"], default="auto",
                    help="gray mode: 1b = improved Model 1 (default if trained), 1 = original")
    ap.add_argument("--dncnn", default=DNCNN_CKPT)
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
        mode = detect_mode(args.image)
        print(f"Auto-detected mode: {mode} "
              f"({'Sentinel-1 file name' if mode == 'sar' else 'not a Sentinel-1 file; use --mode sar to override'})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt = args.checkpoint or default_checkpoint(mode, args.sar_model, args.gray_model)
    encoder, decoder, meta, dncnn = load_models(mode, device, ckpt, args.dncnn,
                                                denoise=not args.no_denoise)
    if meta["input"] == "sar_dncnn_L":
        print("!! --no_denoise: feeding RAW SAR luminance (ablation)" if dncnn is None
              else f"DnCNN loaded from {args.dncnn}")
    name = model_name(meta)
    print(f"Model: {name}  |  checkpoint: {ckpt}  |  encoder fingerprint {meta['fingerprint']}")

    pred, gray_in, pil = colorize(args.image, mode, encoder, decoder, meta, dncnn, device)
    if args.sharp:
        pred = unsharp(pred)
        print("Unsharp mask applied (--sharp)")

    raw_path = os.path.splitext(args.out)[0] + "_raw.png"
    Image.fromarray((pred * 255).round().astype(np.uint8)).save(raw_path)

    gt_path, gt_title = find_ground_truth(args.image, args.root)
    n_panels = 3 if gt_path else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(gray_in, cmap="gray", vmin=0, vmax=1)
    in_label = {"optical_L": "L", "sar3": "SAR"}.get(
        meta["input"], "denoised SAR L" if dncnn is not None else "raw SAR L")
    axes[0].set_title(f"Input ({in_label})")
    axes[1].imshow(pred)
    axes[1].set_title(f"ColorSAR {name} output"
                      f"{' (+unsharp)' if args.sharp else ''}")
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


if __name__ == "__main__":
    main()
