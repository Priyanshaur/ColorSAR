"""
ColorSAR - interactive dashboard (Gradio)

    python app.py        ->  http://127.0.0.1:7860

Three tabs:
    Colorize  upload an image (or pick a dataset sample); Sentinel-1 files (`_s1_` in the
              name) go SAR -> DnCNN -> Model 2, everything else -> Model 1. Shows
              input / colorized / ground truth side by side
    Results   held-out metrics next to the no-color / mean-color baselines,
              per-terrain breakdown, training curves
    About     pipeline, the checkpoint-portability fix and how to reproduce everything

Uses colorize.py's functions directly, so with the same options the dashboard and
`python colorize.py` produce the same image (both default to no sharpening).
"""
import glob
import json
import os
import time

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from color_utils import unsharp  # noqa: E402
from colorize import CHECKPOINTS, colorize, detect_mode, find_ground_truth, load_models  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "v_2"

print(f"[app] device: {DEVICE}   loading models ...")
# every checkpoint that exists, keyed by its display name; first entry per mode = default
MODEL_FILES = {"sar": [("Model 3 (direct SAR -> L + color)", CHECKPOINTS["sar3"]),
                       ("Model 2 (DnCNN -> L -> color, original)", CHECKPOINTS["sar2"])],
               "gray": [("Model 1b (improved)", CHECKPOINTS["gray1b"]),
                        ("Model 1 (original)", CHECKPOINTS["gray"])]}
MODELS, CHOICES, DNCNN = {}, {}, None
for _mode, entries in MODEL_FILES.items():
    CHOICES[_mode] = []
    for _name, _path in entries:
        if not os.path.exists(_path):
            continue
        enc, dec, meta, dn = load_models(_mode, DEVICE, checkpoint=_path)
        MODELS[_name] = (enc, dec, meta)
        CHOICES[_mode].append(_name)
        DNCNN = dn or DNCNN
FINGERPRINT = MODELS[CHOICES["gray"][-1]][2]["fingerprint"]
print(f"[app] ready: {', '.join(MODELS)}")



def _new_format(m):
    """metrics.json written by the current evaluate.py (baselines + eval_set)."""
    return bool(m) and isinstance(m.get("model"), dict) and "eval_set" in m


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


DNCNN_METRICS = _load_json("model2_run/dncnn_metrics.json")


# ---------------------------------------------------------------- helpers
def _samples():
    out = []
    if os.path.isdir(DATA_ROOT):
        for terrain in sorted(os.listdir(DATA_ROOT)):
            files = sorted(glob.glob(os.path.join(DATA_ROOT, terrain, "s1", "*.png")))
            if files:
                out.append((f"{terrain} - SAR (s1)", files[0]))
        for terrain in sorted(os.listdir(DATA_ROOT)):
            files = sorted(glob.glob(os.path.join(DATA_ROOT, terrain, "s2", "*.png")))
            if files:
                out.append((f"{terrain} - optical (s2)", files[0]))
                break
    if os.path.exists("sample_input.jpeg"):
        out.append(("everyday photo (out of domain)", "sample_input.jpeg"))
    return out


SAMPLES = _samples()


def _gray3(a):
    a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
    return (np.stack([a] * 3, axis=-1) * 255).round().astype(np.uint8)


# ---------------------------------------------------------------- tab 1
def run_colorize(upload_path, mode_choice, sar_model, gray_model, denoise, sharpen, sample):
    t0 = time.time()
    try:
        if upload_path:
            path, gt_path = upload_path, None
            src = f"upload `{os.path.basename(upload_path)}`"
        elif sample:
            path = sample
            gt_path, _ = find_ground_truth(sample, DATA_ROOT)
            src = os.path.basename(sample)
        else:
            return [], "Pick a dataset sample or upload an image first."

        mode = detect_mode(path) if mode_choice == "auto" else mode_choice
        chosen = sar_model if mode == "sar" else gray_model
        enc, dec, meta = MODELS[chosen]
        uses_dncnn = meta["input"] == "sar_dncnn_L"
        dncnn = DNCNN if (uses_dncnn and denoise) else None
        pred, gray_in, _ = colorize(path, mode, enc, dec, meta, dncnn, DEVICE)
        notes = []
        if uses_dncnn and not denoise:
            notes.append("DnCNN skipped (ablation: raw SAR luminance)")
        if sharpen:
            pred = unsharp(pred)
            notes.append("output sharpened (unsharp mask, cosmetic)")
        if mode == "gray" and not (gt_path or "_s2_" in os.path.basename(path)):
            notes.append("Model 1 was trained on satellite tiles only - colors on other "
                         "images are a guess in the 'satellite palette'")

        in_label = {"optical_L": "L", "sar3": "SAR"}.get(
            meta["input"], "denoised SAR L" if dncnn is not None else "raw SAR L")
        panels = [(_gray3(gray_in), f"input ({in_label})"),
                  ((pred * 255).round().astype(np.uint8), f"ColorSAR {chosen.split(' (')[0]} output")]
        if gt_path:
            title = "ground truth (s2)" if mode == "sar" else "paired SAR (s1)"
            panels.append((np.array(Image.open(gt_path).convert("RGB")), title))
        else:
            notes.append("no ground truth (only dataset samples have one)")

        info = (f"source: {src}  |  mode: **{mode}**  |  model: **{chosen}**"
                + "".join(f"  \n- {n}" for n in notes)
                + f"  \ncheckpoint: `{meta['checkpoint']}`  |  fingerprint: `{meta['fingerprint']}`"
                  f"  |  device: {DEVICE}  |  {time.time() - t0:.1f}s")
        return panels, info
    except Exception as e:                       # surface errors in the UI, not the console
        return [], f"**Error:** `{type(e).__name__}: {e}`"


# ---------------------------------------------------------------- tab 2
RUNS = {  # key: (run folder, title, upstream PSNR/SSIM or None)
    "1": ("model1_run", "Model 1 - optical L (original recipe)", (29.9230, 0.9400)),
    "1b": ("model1b_run", "Model 1b - optical L, color bins + trained adapters", None),
    "2": ("model2_run", "Model 2 - SAR L via DnCNN (original recipe)", (14.4466, 0.2314)),
    "3": ("model3_run", "Model 3 - SAR tile -> predicted L + color bins", None),
}
METRICS = {k: _load_json(os.path.join(d, "metrics.json")) for k, (d, _, _) in RUNS.items()}
HISTORY = {k: _load_json(os.path.join(d, "history.json")) for k, (d, _, _) in RUNS.items()}
# head-to-head files written by kaggle/run_kaggle.py: both models on identical unseen pairs
H2H = [("Model 2 vs Model 3 (SAR)", "model3_run/metrics_model2_same_pairs.json",
        "model3_run/metrics_vs_model2.json"),
       ("Model 1 vs Model 1b (optical)", "model1b_run/metrics_model1_same_pairs.json",
        "model1b_run/metrics_vs_model1.json")]


def _loss_figure():
    keys = [k for k in RUNS if HISTORY[k]]
    fig, axes = plt.subplots(1, max(len(keys), 1), figsize=(5.2 * max(len(keys), 1), 4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        h = HISTORY[k]
        metric = "val_score" if "val_score" in h else "val_loss"
        ep = range(1, len(h[metric]) + 1)
        best = int(np.argmin(h[metric])) + 1
        if "val_ab_rmse" in h:
            ax.plot(ep, h["val_ab_rmse"], marker="s", ms=3, label="val a/b RMSE")
            if any(h.get("val_L_rmse") or []):
                ax.plot(ep, h["val_L_rmse"], marker="^", ms=3, label="val L RMSE")
            ax.set_ylabel("Lab units")
        else:
            ax.plot(ep, h["train_loss"], marker="o", ms=3, label="train MSE")
            ax.plot(ep, h["val_loss"], marker="s", ms=3, label="val MSE")
            ax.set_ylabel("MSE (normalized a/b)")
        ax.axvline(best, ls=":", c="gray")
        ax.set_title(f"{RUNS[k][1].split(' - ')[0]} - best epoch {best}", fontsize=11)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _terrain_figure():
    keys = [k for k in RUNS if _new_format(METRICS[k])]
    if not keys:
        fig, ax = plt.subplots(figsize=(6, 1)); ax.axis("off")
        ax.set_title("no metrics.json in the new format yet - run evaluate.py"); return fig
    m0 = METRICS[keys[0]]
    terrains = list(m0["per_terrain"])
    x = np.arange(len(terrains)); w = 0.8 / (len(keys) + 1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for j, k in enumerate(keys):
        vals = [METRICS[k]["per_terrain"][t]["model"]["ab_rmse"] for t in terrains]
        b = ax.bar(x + (j - len(keys) / 2) * w, vals, w, label=f"Model {k}")
        ax.bar_label(b, fmt="%.1f", fontsize=7)
    vals = [m0["per_terrain"][t]["mean_color"]["ab_rmse"] for t in terrains]
    b = ax.bar(x + (len(keys) / 2) * w, vals, w, label="mean-color baseline", color="lightgray")
    ax.bar_label(b, fmt="%.1f", fontsize=7)
    ax.set_xticks(x, terrains); ax.set_ylabel("a/b RMSE (Lab units, lower = better)")
    ax.set_title("Color error per terrain (each model on its own held-out set)")
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _row(name, m, r):
    x = m[r]
    return f"| {name} | {x['psnr']:.2f} | {x['ssim']:.4f} | {x['ab_rmse']:.2f} |"


def _results_markdown():
    out = []
    for name, a_path, b_path in H2H:
        a, b = _load_json(a_path), _load_json(b_path)
        if _new_format(a) and _new_format(b):
            ka, kb = name.split(" vs ")[0], name.split(" vs ")[1].split(" (")[0]
            out += [f"#### Head to head: {name} - {a['n']:,} pairs neither model trained on", "",
                    "| | PSNR (dB) | SSIM | a/b RMSE | L RMSE |", "|---|---|---|---|---|",
                    f"| {ka} | {a['model']['psnr']:.2f} | {a['model']['ssim']:.4f} | "
                    f"{a['model']['ab_rmse']:.2f} | {a['L_rmse']:.2f} |",
                    f"| **{kb}** | **{b['model']['psnr']:.2f}** | **{b['model']['ssim']:.4f}** | "
                    f"**{b['model']['ab_rmse']:.2f}** | **{b['L_rmse']:.2f}** |",
                    f"| mean-color baseline ({ka}'s L) | {a['mean_color']['psnr']:.2f} | "
                    f"{a['mean_color']['ssim']:.4f} | {a['mean_color']['ab_rmse']:.2f} | - |", ""]
    for k, (d, title, up) in RUNS.items():
        m = METRICS[k]
        if not os.path.exists(os.path.join(d, f"model_{k[0]}.pth")):
            continue
        out.append(f"#### {title}")
        if not _new_format(m):
            out += [f"*Held-out metrics not available yet* - run `python evaluate.py --model {k[0]} "
                    f"--checkpoint {d}/model_{k[0]}.pth --per_terrain --out {d}/metrics.json`.", ""]
            continue
        out += [f"*Scored on: {m['eval_set']}*", "",
                "| | PSNR (dB) | SSIM | a/b RMSE (Lab) |", "|---|---|---|---|",
                _row("**model**", m, "model"), _row("no-color baseline (a=b=0)", m, "no_color"),
                _row("mean-color baseline", m, "mean_color"),
                _row("oracle: true colors, same L" + (" (round trip)" if k[0] == "1" else ""), m, "oracle")]
        if "L_rmse" in m and k[0] != "1":
            out.append(f"| brightness error (L RMSE) | | | {m['L_rmse']:.2f} |")
        if up:
            out.append(f"| upstream ColorSAR (different split - context only) | {up[0]:.2f} | {up[1]:.4f} | - |")
        out.append("")
    out += ["**How to read this.** PSNR/SSIM on RGB are dominated by luminance. For Models 1/2 "
            "the luminance is *given* (optical L, or the DnCNN's estimate); Model 3 predicts its "
            "own. Color quality is best read from a/b RMSE and the gap to the mean-color "
            "baseline; L RMSE is the brightness error vs the optical image."]
    if DNCNN_METRICS:
        dn, r = DNCNN_METRICS["dncnn"], DNCNN_METRICS["raw_sar"]
        out += ["", f"**DnCNN stage** (SAR gray -> optical gray, {DNCNN_METRICS['n_images']:,} "
                f"held-out pairs): {dn['psnr']:.2f} dB / {dn['ssim']:.4f} SSIM, vs raw SAR "
                f"{r['psnr']:.2f} dB / {r['ssim']:.4f} (upstream reported 16.24 / 0.2641)."]
    return "\n".join(out)


# ---------------------------------------------------------------- app
with gr.Blocks(title="ColorSAR") as demo:
    gr.Markdown("# ColorSAR — colorization from SAR and grayscale imagery")
    gr.Markdown("Reimplementation of [ColorSAR](https://github.com/nishant9083/ColorSAR) "
                "(ensemble ResNet50 + DenseNet121 encoder, U-Net decoder, DnCNN front-end).")

    with gr.Tabs():
        with gr.Tab("Colorize"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="filepath", image_mode=None,
                                   label="Upload your own image (optional)")
                    gr.Markdown("*Wait for the preview to appear before clicking Colorize — "
                                "otherwise the dataset sample below is used instead.*")
                    sample = gr.Dropdown(choices=SAMPLES, value=SAMPLES[0][1] if SAMPLES else None,
                                         label="...or pick a sample")
                    mode_choice = gr.Radio(
                        ["auto", "sar", "gray"], value="auto", label="Input type",
                        info="auto: Sentinel-1 file name (`_s1_`) -> SAR model, anything else "
                             "-> optical model. Choose 'sar' for other SAR images.")
                    sar_model = gr.Dropdown(CHOICES["sar"], value=CHOICES["sar"][0],
                                            label="Model for SAR inputs")
                    gray_model = gr.Dropdown(CHOICES["gray"], value=CHOICES["gray"][0],
                                             label="Model for optical / grayscale inputs")
                    denoise = gr.Checkbox(value=True,
                                          label="Apply DnCNN before colorizing (Model 2 only)")
                    sharpen = gr.Checkbox(value=False,
                                          label="Sharpen output (unsharp mask, cosmetic)")
                    btn = gr.Button("Colorize", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="input | ColorSAR output | ground truth (dataset samples only)",
                                         columns=3, height=420, object_fit="contain")
                    info = gr.Markdown()
            btn.click(run_colorize, [img, mode_choice, sar_model, gray_model, denoise, sharpen, sample],
                      [gallery, info])

        with gr.Tab("Results"):
            gr.Markdown(_results_markdown())
            gr.Plot(_terrain_figure())
            gr.Plot(_loss_figure())

        with gr.Tab("About"):
            gr.Markdown(f"""
**Pipeline**

| input | path | checkpoint |
|---|---|---|
| Sentinel-1 SAR tile, Model 3 | s1 (3 despeckling levels) -> encoder-decoder -> predicted L + color | `{CHECKPOINTS['sar3']}` |
| Sentinel-1 SAR tile, Model 2 | s1 -> DnCNN -> L channel -> colorizer | `{CHECKPOINTS['sar2']}` |
| optical image, Model 1 / 1b | L of the input -> colorizer | `{CHECKPOINTS['gray']}` / `{CHECKPOINTS['gray1b']}` |

Models that are not trained yet simply do not appear in the dropdowns.

The encoder is a frozen ImageNet ResNet50 + DenseNet121 ensemble with 1x1 projections
and fusion blocks ("adapters"); only the U-Net decoder is trained (MSE on Lab a/b).
Color is predicted at 224x224 and combined with the input's full-resolution luminance.
Both models were trained on 256x256 Sentinel-2 tiles only.

**Checkpoint portability (the bug we found and fixed)**

The adapters have no pretrained weights. The shipped checkpoints store only the decoder,
so the adapters must be rebuilt identically everywhere — originally nothing seeded them,
and `ResNet50_Weights.DEFAULT` silently flipped V1->V2 across torchvision releases. Result:
shipped checkpoints scored **0.045 val MSE instead of the 0.0043** they were trained to.

Fix: `ENCODER_FROZEN_SEED = 42`, pinned `IMAGENET1K_V2` weights, and loading a decoder-only
checkpoint now *fails* unless the rebuilt adapters match fingerprint **`{FINGERPRINT}`**.
New training runs (`train_colorizer.py`) save the adapters inside the checkpoint, so they
no longer depend on the RNG at all.

**Reproduce**

```powershell
python train_dncnn.py                 # stage A: SAR-gray -> optical-gray network
python Train_model2.py                # stage B: Model 2 colorizer
python evaluate.py --model 2 --per_terrain
python colorize.py --image v_2/agri/s1/<file>.png     # CLI demo
.\\run_demo.ps1                       # full scripted demo
```
""")

if __name__ == "__main__":
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)
    demo.launch(server_name="127.0.0.1", server_port=port,
                inbrowser=not os.environ.get("COLORSAR_NO_BROWSER"),
                show_error=True, theme=gr.themes.Soft())
