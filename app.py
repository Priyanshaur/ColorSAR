"""
ColorSAR - interactive dashboard (Gradio)

    python app.py        ->  http://127.0.0.1:7860

Three tabs:
    Colorize  upload an image (or pick a dataset sample), auto-detects whether it is
              SAR (-> DnCNN -> Model 2) or a photo (-> Model 1) and shows
              input / colorized / ground-truth side by side
    Results   final metrics vs the upstream ColorSAR numbers, per-terrain breakdown,
              training curves
    About     pipeline, the portability fix and how to reproduce everything

Reuses the exact same code path as the CLI demo (colorize.py) -- nothing is duplicated,
so what you see here is what `python colorize.py` produces.
"""
import glob
import json
import os
import tempfile
import time

import gradio as gr
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

matplotlib.use("Agg")

from colorize import colorize, find_ground_truth          # noqa: E402  (CLI, guarded)
from color_utils import load_decoder                      # noqa: E402
from dncnn import load_dncnn                              # noqa: E402
from model_architecture import EnsembleEncoder, encoder_fingerprint  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M1_CKPT = "model1_run/model_1.pth"
M2_CKPT = "model2_run/model_2.pth"
DNCNN_CKPT = "model2_run/dncnn.pth"

assert os.path.exists(M1_CKPT) and os.path.exists(M2_CKPT), "checkpoints missing"
assert os.path.exists(DNCNN_CKPT), "DnCNN checkpoint missing"

print(f"[app] device: {DEVICE}   loading models ...")
ENCODER = EnsembleEncoder(pretrained=True).to(DEVICE).eval()
for p in ENCODER.parameters():
    p.requires_grad = False
DECODERS = {"sar": load_decoder(M2_CKPT, DEVICE), "gray": load_decoder(M1_CKPT, DEVICE)}
DNCNN = load_dncnn(DNCNN_CKPT, DEVICE)
FINGERPRINT = encoder_fingerprint(ENCODER)
print(f"[app] ready. encoder fingerprint: {FINGERPRINT}")

UPSTREAM = {"m1_psnr": 29.9230, "m1_ssim": 0.9400, "m2_psnr": 14.4466, "m2_ssim": 0.2314}


# ---------------------------------------------------------------- helpers
def _samples():
    out = []
    for terrain in sorted(os.listdir("v_2")):
        files = sorted(glob.glob(os.path.join("v_2", terrain, "s1", "*.png")))
        if files:
            out.append((f"{terrain} - SAR (s1)", files[0]))
    if os.path.exists("sample_input.jpeg"):
        out.append(("photo - Model 1 input", "sample_input.jpeg"))
    return out


SAMPLES = _samples()


def _to_temp(pil_img):
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_img.save(f.name)
    return f.name


def _gray3(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.max() <= 1.0:
        a = (a * 255).astype(np.uint8)
    return a.astype(np.uint8)


def _looks_grayscale(pil_img):
    """Gradio normalizes uploads to RGB, so a 1-band SAR file arrives as R=G=B.
    Detect that case from the pixel content instead of the band count."""
    a = np.asarray(pil_img.convert("RGB"), dtype=np.int16)
    return (np.abs(a[..., 0] - a[..., 1]).max() <= 2
            and np.abs(a[..., 1] - a[..., 2]).max() <= 2)


# ---------------------------------------------------------------- tab 1
def run_colorize(img, mode_choice, denoise, sample):
    t0 = time.time()
    tmp = None
    try:
        if img is not None:
            tmp = _to_temp(img)
            path, gt_path, gt_title = tmp, None, None
            src = "uploaded image"
        elif sample:
            path = sample
            gt_path, gt_title = find_ground_truth(sample, "v_2")
            src = os.path.basename(sample)
        else:
            return [], "Pick a dataset sample or upload an image first."

        mode = mode_choice
        if mode == "auto":
            if img is not None:
                # uploads arrive RGB-normalized: judge by pixel content instead
                mode = "sar" if _looks_grayscale(img) else "gray"
            else:
                bands = len(Image.open(path).getbands())
                mode = "sar" if bands == 1 else "gray"

        dncnn = DNCNN if (mode == "sar" and denoise) else None
        if mode == "sar" and not denoise:
            note = " | DnCNN skipped (ablation: raw SAR luminance)"
        else:
            note = ""

        pred, gray_in, pil = colorize(path, mode, ENCODER, DECODERS[mode], dncnn, DEVICE)

        panels = [(_gray3(gray_in), f"input ({'denoised SAR L' if dncnn else 'L'})"),
                  ((pred * 255).astype(np.uint8),
                   f"ColorSAR {'Model 2' if mode == 'sar' else 'Model 1'} output")]
        if gt_path:
            panels.append((np.array(Image.open(gt_path).convert("RGB")), "ground truth"))
            panels_note = ""
        else:
            panels_note = " — no ground truth (only dataset samples have one; your upload's true colors are unknown)"

        info = (f"source: {src}  |  mode: **{mode}**  |  model: "
                f"{'2 (SAR -> DnCNN -> L -> colorizer)' if mode == 'sar' else '1 (L -> colorizer)'}"
                f"{note}{panels_note}\ncheckpoint: `{M2_CKPT if mode == 'sar' else M1_CKPT}`  |  "
                f"fingerprint: `{FINGERPRINT}`  |  device: {DEVICE}  |  "
                f"{time.time() - t0:.1f}s")
        return panels, info
    except Exception as e:                       # surface errors in the UI, not the console
        return [], f"**Error:** `{type(e).__name__}: {e}`"
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------- tab 2
def _loss_figure():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (f, title) in zip(axes, [("model1_run/history.json", "Model 1 - 20 epochs (Kaggle, 2x T4)"),
                                     ("model2_run/history.json", "Model 2 - 15 epochs (laptop, RTX 4050)")]):
        h = json.load(open(f))
        ax.plot(h["train_loss"], marker="o", ms=3, label="train")
        ax.plot(h["val_loss"], marker="s", ms=3, label="val")
        ax.set_title(f"{title}\nbest val {min(h['val_loss']):.5f}", fontsize=11)
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE (ab space)")
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def _terrain_figure():
    m1 = json.load(open("model1_run/metrics.json"))
    m2 = json.load(open("model2_run/metrics.json"))
    terrains = list(m1["per_terrain"].keys())
    x = np.arange(len(terrains)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.2))
    b1 = ax.bar(x - w / 2, [m1["per_terrain"][t]["psnr"] for t in terrains], w, label="Model 1 (optical L)")
    b2 = ax.bar(x + w / 2, [m2["per_terrain"][t]["psnr"] for t in terrains], w, label="Model 2 (SAR L)")
    ax.axhline(UPSTREAM["m1_psnr"], ls="--", c="tab:blue", lw=1, label="upstream M1 (29.92)")
    ax.axhline(UPSTREAM["m2_psnr"], ls="--", c="tab:orange", lw=1, label="upstream M2 (14.45)")
    ax.bar_label(b1, fmt="%.1f", fontsize=8); ax.bar_label(b2, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, terrains); ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR per terrain - full 16,000-pair evaluation")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def _results_markdown():
    m1 = json.load(open("model1_run/metrics.json"))
    m2 = json.load(open("model2_run/metrics.json"))
    return f"""
| Model | our PSNR | our SSIM | upstream ColorSAR | oracle ceiling |
|---|---|---|---|---|
| **Model 1** (optical L) | **{m1['psnr']:.4f}** | **{m1['ssim']:.4f}** | {UPSTREAM['m1_psnr']:.4f} / {UPSTREAM['m1_ssim']:.4f} | {m1['oracle_psnr']:.2f} dB |
| **Model 2** (SAR L, DnCNN) | **{m2['psnr']:.4f}** | **{m2['ssim']:.4f}** | {UPSTREAM['m2_psnr']:.4f} / {UPSTREAM['m2_ssim']:.4f} | **{m2['oracle_psnr']:.4f} dB** |

- Evaluation on all **{m1['n_images']:,}** paired s1/s2 images, after retraining with the
  portable (seeded) encoder — fingerprint `{FINGERPRINT}`.
- **Model 2 reaches {100 * m2['psnr'] / m2['oracle_psnr']:.1f}% of its theoretical ceiling**:
  the limit is the SAR luminance it is given, not the colorizer.
- Scores from the broken (pre-fix) checkpoints were 10-50x worse — see the About tab.
"""


# ---------------------------------------------------------------- app
with gr.Blocks(title="ColorSAR", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ColorSAR — colorization from SAR and grayscale imagery")
    gr.Markdown("Faithful reimplementation of [ColorSAR](https://github.com/nishant9083/ColorSAR) "
                "(ensemble ResNet50 + DenseNet121 encoder, U-Net decoder, DnCNN front-end).")

    with gr.Tabs():
        with gr.Tab("Colorize"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="pil", label="Upload your own image (optional)")
                    gr.Markdown("*Wait for the preview to appear before clicking Colorize — "
                                "otherwise the dataset sample below is used instead.*")
                    sample = gr.Dropdown(choices=SAMPLES, value=SAMPLES[0][1],
                                         label="...or pick a dataset sample")
                    mode_choice = gr.Radio(["auto", "sar", "gray"], value="auto",
                                           label="Input type",
                                           info="auto: grayscale-looking -> SAR (Model 2), color -> photo (Model 1). "
                                                "Set 'sar' manually if a grayscale photo is misdetected.")
                    denoise = gr.Checkbox(value=True,
                                          label="Apply DnCNN denoising before colorizing (SAR only)")
                    btn = gr.Button("Colorize", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="input | ColorSAR output  (ground truth only for dataset samples)",
                                         columns=3, height=420, object_fit="contain")
                    info = gr.Markdown()
            btn.click(run_colorize, [img, mode_choice, denoise, sample], [gallery, info])

        with gr.Tab("Results"):
            gr.Markdown(_results_markdown())
            with gr.Row():
                gr.Plot(_terrain_figure())
            gr.Plot(_loss_figure())

        with gr.Tab("About"):
            gr.Markdown(f"""
**Pipeline**

| input | path | checkpoint |
|---|---|---|
| SAR image (s1, 1 band) | s1 -> DnCNN denoise -> L channel -> colorizer | `{M2_CKPT}` |
| photo / optical (3 band) | L of the input -> colorizer | `{M1_CKPT}` |

The encoder is a frozen ImageNet ResNet50 + DenseNet121 ensemble with learned 1x1
projections and fusion blocks; only the U-Net decoder is trained (MSE in Lab a/b).

**Checkpoint portability (the bug we found and fixed)**

The encoder's `conv1x1_*` / `fusion_blocks` layers have no pretrained weights: they start
random and stay frozen. Checkpoints store only the decoder, so those layers must be
rebuilt identically everywhere — nothing seeded them, and `ResNet50_Weights.DEFAULT`
silently flipped V1->V2 across torchvision releases. Result: shipped checkpoints scored
**0.045 val MSE instead of 0.0043** they were trained to (10-50x worse).

Fix: `ENCODER_FROZEN_SEED = 42`, pinned `IMAGENET1K_V2` weights, and a fingerprint printed
by every script: **`{FINGERPRINT}`**. Training and evaluation must print the same value —
the numbers on the Results tab were produced with exactly this encoder.

**Reproduce**

```powershell
python train_dncnn.py                 # stage A: SAR-gray -> optical-gray denoiser
python Train_model2.py                # stage B: Model 2 colorizer
python evaluate.py --model 2 --checkpoint {M2_CKPT} --per_terrain
python colorize.py --image v_2/agri/s1/<file>.png     # CLI demo
.\\run_demo.ps1                       # full scripted demo
```
""")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                show_error=True)
