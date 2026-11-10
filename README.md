# ColorSAR — Synthetic Aperture Radar (SAR) Image Colorization

An end-to-end deep learning framework for colorizing grayscale satellite imagery (Sentinel-1 SAR / Sentinel-2 optical pairs) using an **Ensemble Encoder-Decoder** architecture in the CIE $L^*a^*b^*$ color space.

---

## 1. Project Overview & Architecture

ColorSAR converts single-channel luminance ($L^*$, brightness) into realistic two-channel chrominance ($a^*, b^*$, color), reconstructing natural RGB satellite imagery.

```
                      +-------------------+
                      |   L Channel (1)   |
                      |   (repeat to 3ch) |
                      +---------+---------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
     +---------------+                     +---------------+
     |   ResNet-50   |                     | DenseNet-121  |
     | (Pretrained)  |                     | (Pretrained)  |
     +-------+-------+                     +-------+-------+
             |                                     |
             +------------------+------------------+
                                |
                                v
                     +--------------------+
                     |  1x1 Convolutions  |
                     |  & Fusion Blocks   |
                     +---------+----------+
                               |
                   Skip Connections (56, 28, 14, 7)
                               |
                               v
                     +--------------------+
                     |    U-Net Style     |
                     |      Decoder       |
                     +---------+----------+
                               |
                               v
                     Predicted (a, b) Channels
                               |
                 Combine with L -> Reconstructed RGB
```

- **Ensemble Encoder**: Combines feature extractors from **ResNet-50** and **DenseNet-121** (ImageNet-pretrained, frozen during Model 1 training). Intermediate multi-scale features are fused using $1 \times 1$ convolutions and batch-normalized fusion blocks.
- **U-Net Decoder**: Transposed convolutions and bilinear upsampling blocks with skip connections restore spatial resolution from $7 \times 7$ back to $224 \times 224 \times 2$.
- **Total Model Parameters**: `45,597,958` (~45.6M).

---

## 2. Model 1 Training & Results (`model1_run/`)

Model 1 (clean-input baseline colorization) was trained across the **full dataset** (~16,000 paired satellite images across all 4 terrain classes: Agricultural, Urban, Barren, and Grassland).

### Training Configuration
- **Hardware**: Kaggle 2x NVIDIA Tesla T4 GPUs (31.2 GB total VRAM) with `torch.nn.DataParallel`
- **Dataset**: `requiemonk/sentinel12-image-pairs-segregated-by-terrain` (v_2)
- **Batch Size**: 64 (32 per GPU)
- **Epochs**: 20
- **Optimizer**: Adam ($\text{lr} = 5 \times 10^{-4}$ with `CosineAnnealingLR` scheduler)
- **Loss Function**: Mean Squared Error (MSE) on normalized $a^*, b^*$ channels

### Convergence & Loss Progression

| Epoch | Train Loss (MSE) | Validation Loss (MSE) | Notes |
| :---: | :---: | :---: | :--- |
| **1** | `0.0384` | `0.0129` | Initial color distribution convergence |
| **5** | `0.0087` | `0.0079` | Rapid reduction in loss |
| **10** | `0.0108` | `0.0075` | Generalization across diverse terrain sets |
| **15** | `0.0060` | `0.0051` | Stable fine-tuning of chromatic details |
| **17** | `0.0056` | **`0.0043`** | **Best Validation Loss reached** |
| **20** | **`0.0049`** | `0.0043` | Final converged checkpoint saved |

### Artifacts in `model1_run/`
- **`model_1.pth`**: Trained PyTorch decoder weights (31.3 MB).
  ⚠️ This file is **not in the repo yet** — the old `*.pth` ignore rule stripped it
  out of the training commit. `.gitignore` now has an exception, so it just needs
  `git add -f model1_run/model_1.pth && git push` from the machine that trained it.
  `colorize_image.py` and `evaluate.py` both need it.
- **[`loss_curve.png`](model1_run/loss_curve.png)**: 20-epoch training vs. validation loss curve.
- **[`sample_predictions.png`](model1_run/sample_predictions.png)**: 3-panel visual evaluation (Grayscale Input vs. Ground Truth RGB vs. ColorSAR Prediction).
- **[`history.json`](model1_run/history.json)**: Raw JSON loss logs for each epoch.
- **[`colorized_sample_test.png`](model1_run/colorized_sample_test.png)**: Local inference verification output on `sample_input.jpeg`.

---

## 3. Local Environment & Hardware Verification

The project is configured and verified to run locally on your laptop:

- **Laptop Hardware**:
  - **CPU**: Intel Core i5 (13th Gen)
  - **RAM**: 24 GB DDR5
  - **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, CUDA 12.6 / Driver 581.86)
- **Virtual Environment**: `.venv`
- **Key Installed Packages**:
  - `torch==2.14.0+cu126` & `torchvision==0.29.0+cu126` (GPU enabled)
  - `opencv-python==5.0.0.93`
  - `scikit-image==0.26.0`
  - `matplotlib==3.11.2`
  - `pillow==12.3.0`
  - `numpy==2.5.3`

---

## 4. How to Run Local Inference / Colorization

Use [`colorize_image.py`](colorize_image.py) to test the trained model on any image:

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Colorize a sample image using your trained model
python colorize_image.py --image sample_input.jpeg --checkpoint model1_run/model_1.pth --out colorized_result.png
```

This will run inference on your RTX 4050 GPU and generate a 3-way side-by-side comparison:
1. Grayscale $L$ Input
2. ColorSAR Predicted Color
3. Ground Truth Original Image

---

## 5. Repository File Structure

```
ColorSAR/
├── model1_run/                    # Model 1 (trained): checkpoint + results
│   ├── model_1.pth                # 31.3 MB decoder weights (see note in section 2)
│   ├── loss_curve.png             # 20-epoch convergence plot
│   ├── sample_predictions.png     # Validation sample comparisons
│   ├── history.json               # Epoch-by-epoch loss records
│   ├── metrics.json               # Final PSNR/SSIM + per-terrain breakdown
│   └── colorized_sample_test.png  # Local verification test on sample_input.jpeg
├── model2_run/                    # Model 2 (trained): DnCNN + colorizer + results
│   ├── dncnn.pth                  # Trained denoiser weights
│   ├── model_2.pth                # Trained decoder weights
│   ├── history.json / metrics.json # Loss records + final PSNR/SSIM
│   └── sample_predictions.png     # Validation sample comparisons
├── ColorSAR_Kaggle_Training.ipynb # Multi-GPU training notebook for Kaggle (2x T4)
├── colorize.py                    # Unified inference: SAR -> DnCNN -> Model 2, photo -> Model 1
├── app.py                         # Interactive Gradio dashboard (Colorize / Results / About)
├── run_demo.ps1                   # One-command scripted demo (opens all figures)
├── colorize_image.py              # Standalone inference & colorization CLI tool
├── evaluate.py                    # PSNR / SSIM evaluation for Model 1 or Model 2
├── Train_model2.py                # Stage B: colorization trained on SAR input
├── train_dncnn.py                 # Stage A: DnCNN denoiser training
├── dncnn.py                       # DnCNN architecture (depth 17, residual learning)
├── color_utils.py                 # s1/s2 pairing + Lab conversion helpers
├── model_architecture.py          # EnsembleEncoder, Decoder & ColorizationModel
├── data_pipeline.py               # Lab space conversion & patch extraction logic
├── Train_model1.py                # Local training script (supports subset via --n_images)
├── visual_test.py                 # Pipeline sanity check script
├── requirements.txt               # Dependencies file
├── sample_input.jpeg              # Sample test satellite image
└── README.md                      # Complete project documentation
```

---

## 6. Model 2 Pipeline — DnCNN denoising + SAR-input colorization

Model 2 is the *real* SAR colorization path: the luminance fed to the network comes
from the **SAR (s1)** image after DnCNN denoising, while the a/b targets stay the
optical (s2) channels. This reproduces `gray_model.ipynb` from the original repo
(DnCNN depth 17 / 64 channels, residual learning; encoder frozen; MSE; Adam 1e-3).

**Stage A — train the denoiser** (SAR grayscale → optical grayscale):

```powershell
python train_dncnn.py --epochs 15
```
Writes `model2_run/dncnn.pth`, its loss curve/history, and prints the denoiser's own
PSNR/SSIM (original reported **16.24 dB / 0.2641**).

**Stage B — train Model 2** on denoised SAR luminance:

```powershell
python Train_model2.py
```
Writes `model2_run/model_2.pth` (best decoder), `history.json`, `loss_curve.png` and
`sample_predictions.png`.

Useful flags for both: `--n_images`, `--epochs`, `--batch_size`, `--out`, `--root`.
`Train_model2.py --no_denoise` trains on raw SAR luminance (ablation for the report).
The encoder is frozen, so `--batch_size 32` fits comfortably in the RTX 4050's 6 GB.

---

## 7. Evaluation — PSNR / SSIM

`evaluate.py` reconstructs RGB from (input L + predicted ab) and scores it against the
paired optical image:

```powershell
python evaluate.py --model 1 --checkpoint model1_run/model_1.pth --per_terrain
python evaluate.py --model 2 --checkpoint model2_run/model_2.pth --per_terrain --out model2_run/metrics.json
```

It prints overall PSNR/SSIM, an **oracle** row (true ab with the same L — the ceiling
imposed by that luminance), and an optional per-terrain breakdown.

### Results (retrained with the portable encoder, fingerprint `788e0728e71f`)

Full-dataset evaluation, 16,000 s1↔s2 pairs (`n_images 16000`):

| Model | our PSNR | our SSIM | upstream target | oracle ceiling |
|---|---|---|---|---|
| **Model 1** (optical L) | **28.4202** | **0.9277** | 29.9230 / 0.9400 | 128.37 |
| **Model 2** (SAR L, DnCNN) | **14.3632** | **0.1952** | 14.4466 / 0.2314 | 14.5596 |

Model 2 sits **0.08 dB below upstream and 0.20 dB below the theoretical ceiling** set by
the DnCNN-smoothed luminance it is given — i.e. it extracts essentially all the color
information that luminance contains. Model 1 is 1.5 dB below upstream, with terrain
spread 31.51 dB (barrenland) → 26.33 dB (grassland).

Per-terrain (both models): see `model1_run/metrics.json` and `model2_run/metrics.json`.

---

## 8. Demo — colorize any image

`colorize.py` is the end-to-end entry point. It auto-detects the input type and picks
the matching model:

| input | path taken | checkpoint |
|---|---|---|
| 1-band SAR image (`v_2/*/s1/*.png`) | s1 → **DnCNN** → L → Model 2 | `model2_run/model_2.pth` |
| RGB / grayscale photo | L → Model 1 | `model1_run/model_1.pth` |

```powershell
python colorize.py --image v_2/agri/s1/ROIs1868_summer_s1_59_p10.png
python colorize.py --image sample_input.jpeg --mode gray
```

**Or run the whole demo in one command** (opens all result figures at the end):

```powershell
.\run_demo.ps1          # SAR + urban + photo + live evaluation (~3-4 min on CPU)
.\run_demo.ps1 -Fast    # visuals only (~30 s)
```

### Interactive dashboard (optional, for presentations)

```powershell
python app.py           # -> http://127.0.0.1:7860 (opens automatically)
```

Three tabs: **Colorize** (upload your own image or pick a dataset sample — auto-detects
SAR vs photo, shows input / output / ground truth, ~3 s per image), **Results** (metrics
vs upstream, per-terrain bars, training curves) and **About** (pipeline + the
portability-fix story). It reuses the exact `colorize.py` code path, so the dashboard
and the CLI always agree.

It writes `<out>` (side-by-side figure: input / colorized / ground truth when the file
belongs to `v_2/`) and `<out stem>_raw.png` (the colorized image at full resolution).

### Checkpoint portability (important)

The encoder's `conv1x1_*` and `fusion_blocks` layers have **no pretrained weights** —
they are random at init and then frozen. Checkpoints store only the decoder, so those
layers must be reconstructed identically on every machine, otherwise the decoder is fed
features it has never seen and the metrics collapse (measured: **0.045 val MSE instead
of the 0.0043 it was trained to**).

`model_architecture.py` now seeds them (`ENCODER_FROZEN_SEED`) and pins
`ResNet50_Weights.IMAGENET1K_V2` (the `DEFAULT` enum silently flipped V1→V2 across
torchvision releases). Every script prints

```
Encoder frozen-layer fingerprint: 788e0728e71f
```

**Training and evaluation must print the same fingerprint** — if they don't, don't
trust the numbers you are about to compute.

---

## 9. Status & Next Steps

Done:

- ✅ **Both models retrained** with the seeded/pinned encoder (Model 1 on Kaggle 2×T4,
  Model 2 on the laptop; commit `9617c26` + `a93af75`).
- ✅ **Both scored on the full 16,000-pair set** — see the results table in §7.
- ✅ **Portability verified**: a fresh process on a different machine reproduces the
  checkpoints' recorded validation losses (Model 1 ≈ 0.0044, Model 2 ≈ 0.0061) and
  prints the same encoder fingerprint `788e0728e71f`.
- ✅ **Demo**: `python colorize.py --image <any s1 png>` — one entry point for both models.

Remaining:

- **Report / slides**: results tables (§7), loss curves (`model1_run/loss_curve.png`,
  `model2_run/loss_curve.png`), sample predictions, and the demo figure from `colorize.py`.
- **Optional ablation**: `Train_model2.py --no_denoise` (raw SAR luminance, no DnCNN)
  for a one-row comparison in the report.
