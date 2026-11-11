# ColorSAR — colorizing SAR and grayscale satellite imagery

Deep-learning colorization of Sentinel-1 SAR and grayscale Sentinel-2 imagery in the CIE
$L^*a^*b^*$ color space.

```
          input (3 channels)
     +--------------+--------------+
     v                             v
 ResNet-50 (frozen)        DenseNet-121 (frozen)
     |                             |
  1x1 projections + fusion blocks ("adapters", 4 scales)
                   |
   U-Net decoder with skip connections 56/28/14/7
                   |
   a/b (and for Model 3 also L) at 224x224  ->  Lab  ->  RGB
```

- **Backbones**: ImageNet ResNet-50 (`IMAGENET1K_V2`) + DenseNet-121 (`IMAGENET1K_V1`), always frozen (30.5 M parameters).
- **Adapters**: 1×1 projections + fusion blocks (7.3 M). Random and frozen in the original recipe; trained in Models 1b and 3.
- **Decoder**: U-Net, bilinear upsampling (7.8 M).

## The four models

| | input | predicts | a/b objective | adapters | checkpoint |
|---|---|---|---|---|---|
| **Model 1** (original) | L of the optical (s2) tile | a/b | MSE | random, frozen | `model1_run/model_1.pth` |
| **Model 1b** | L of the optical tile | a/b | color bins | trained | `model1b_run/model_1.pth` |
| **Model 2** (original) | SAR tile → DnCNN → L | a/b | MSE | random, frozen | `model2_run/model_2.pth` (+ `dncnn.pth`) |
| **Model 3** | the SAR tile itself | **L and a/b** | color bins + L1 on L | trained | `model3_run/model_3.pth` |

What changed in 1b / 3, and why:
- **Color-bin classification** (Zhang et al. 2016, *Colorful Image Colorization*) instead of MSE:
  the network outputs a probability over 152 occupied 10×10 a/b cells, trained with
  class-rebalanced cross-entropy on soft targets and decoded with the annealed mean (T = 0.38).
  MSE regresses to the *average* plausible color, which is why the original outputs look
  gray-green; classification commits to a color.
- **Trained adapters + ImageNet input normalization**: in the original recipe the adapters
  are a random projection of the backbone features and the backbones receive L in [-1, 1],
  unlike anything they were pretrained on.
- **Model 3 reads the SAR tile directly** (raw + two Gaussian-smoothed versions as the three
  channels) and **predicts brightness as well as color**. Model 2 gets its brightness from
  the DnCNN, which caps it: even perfect colors on that brightness only reach 14.56 dB.
- **Rotation/flip augmentation** (satellite tiles have no "up").

Data: `requiemonk/sentinel12-image-pairs-segregated-by-terrain`, folder `v_2/` — 16,000
SAR/optical pairs, 256×256, 4,000 per terrain (agri, barrenland, grassland, urban). All
models were trained on satellite tiles only.

---

## 1. Results (held-out data only)

Every score below is on pairs the model never trained on. Each model is shown next to two
baselines that use **the same brightness (L)**: *no color* (a = b = 0) and *mean color*
(one constant a/b — the average training color). RGB PSNR/SSIM are dominated by brightness,
so for color quality read **a/b RMSE** (color error in Lab units, lower = better) and the
gap to the baselines. **L RMSE** is the brightness error vs the optical image.

### Summary

| model | test pairs | PSNR (dB) | SSIM | a/b RMSE | L RMSE | mean-color baseline a/b RMSE |
|---|---|---|---|---|---|---|
| Model 1 (original) | 1,600 val¹ | 28.42 | 0.928 | 7.52 | 0 (given) | 10.11 |
| **Model 1b** | 1,630 test | **30.02** | **0.935** | **6.15** | 0 (given) | 10.02 |
| Model 2 (original) | 6,000 test | 14.37 | 0.195 | 9.16 | 20.5³ | 10.12 |
| **Model 3** | 1,630 test | **16.62** | **0.282** | **7.56** | **15.77** | 10.02 |
| *upstream ColorSAR, Model 1 / Model 2* | *unknown split²* | *29.92 / 14.45* | *0.940 / 0.231* | – | – | – |

¹ The original Model 1 trained on 90 % of all pairs and validated on the rest, so only its
validation pairs are unseen (and they picked the best epoch: slightly optimistic).
² Reported by the original project; different data selection and split, so context only.  
³ Measured on the 633-pair head-to-head set below.

### Head to head on identical pairs

Different models have different held-out sets, so each improved model was also scored
against its original on **exactly the same pairs that neither of them trained on**:

| pairs | model | PSNR (dB) | SSIM | a/b RMSE | L RMSE |
|---|---|---|---|---|---|
| 633 SAR pairs | Model 2 | 14.30 | 0.196 | 9.10 | 20.51 |
| | **Model 3** | **16.64** | **0.282** | **7.61** | **15.69** |
| 171 optical pairs | Model 1 | 28.92 | 0.931 | 7.18 | 0 |
| | **Model 1b** | **30.38** | **0.939** | **5.91** | 0 |

### Per terrain (a/b RMSE; mean-color baseline in brackets)

| terrain | Model 1 | Model 1b | Model 2 | Model 3 |
|---|---|---|---|---|
| agri | 7.77 (8.95) | 6.66 (8.98) | 9.55 (9.16) | 8.55 (8.98) |
| barrenland | 4.85 (6.93) | 4.03 (7.13) | 6.85 (7.11) | 4.99 (7.13) |
| grassland | 9.31 (16.78) | 7.59 (16.21) | 12.27 (16.60) | 9.70 (16.21) |
| urban | 8.24 (7.73) | 6.12 (7.66) | 8.01 (7.68) | 6.74 (7.66) |

The originals lose to the constant-color baseline on some terrains (Model 1 on urban,
Model 2 on agri, barren and urban). **Models 1b and 3 beat it on every terrain.**

### DnCNN stage of Model 2 (6,000 held-out pairs)

| | PSNR (dB) | SSIM |
|---|---|---|
| DnCNN output vs optical gray | 15.95 | 0.237 |
| raw SAR, no network | 9.55 | 0.023 |
| upstream reported | 16.24 | 0.264 |

### What the numbers say

- **Model 1b is the best optical colorizer**: 18 % less color error than Model 1 on the same
  pairs, 30.0 dB PSNR (above the original project's 29.92), and visibly closer to the ground
  truth (`model1b_run/sample_predictions.png`).
- **Model 3 is a large step for SAR**: +2.3 dB PSNR, +0.09 SSIM, 16 % less color error and 23 % less brightness error than
  Model 2, beating the ceiling that bounded *any* colorizer on Model 2's brightness
  (14.56 dB). In its outputs fields, bare soil, forest and built-up areas get distinct, correct
  color families — Model 2's are uniformly gray-green.
- **Limits that remain**: Model 3 is blurry — its brightness is inferred from radar at
  224×224 and SAR simply does not contain optical detail. Its L error (15.8 Lab units) is
  still the main thing keeping PSNR in the 16-17 dB range. See §7.

Full numbers: `model*_run/metrics*.json`, and the dashboard's *Results* tab.

---

## 2. Setup

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# PyTorch first, matching your machine (GPU build shown; use .../whl/cpu for CPU only):
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

All versions are pinned in [`requirements.txt`](requirements.txt). Put the dataset at `v_2/`.

---

## 3. Colorize an image / demo

[`colorize.py`](colorize.py) is the single inference path (the dashboard and helper scripts use it).
It uses the best available model: **Model 3 for SAR, Model 1b for optical** (falling back to
2 / 1 if those checkpoints are absent).

```powershell
python colorize.py --image v_2/agri/s1/ROIs1868_summer_s1_59_p10.png   # SAR -> Model 3
python colorize.py --image v_2/agri/s1/ROIs1868_summer_s1_59_p10.png --sar_model 2   # original pipeline
python colorize.py --image v_2/agri/s2/ROIs1868_summer_s2_59_p10.png   # optical -> Model 1b
python colorize.py --image some_grayscale_photo.jpg --gray_model 1     # original Model 1
python colorize.py --image other_sar_image.png --mode sar              # SAR without an _s1_ name
```

- **Mode detection** (`--mode auto`): SAR only if the file is a Sentinel-1 tile by name
  (`_s1_` in the file name, or inside an `s1/` folder); everything else — RGB or grayscale —
  is treated as optical. Override with `--mode sar|gray`.
- **Resolution**: color is predicted at 224×224 and upsampled. Models 1/1b/2 combine it with
  the input's full-resolution luminance; Model 3 upsamples its own predicted luminance.
- **Ground truth**: for a `v_2/` file its pair is shown as a third panel.
- **Out-of-domain inputs**: an everyday photo such as `sample_input.jpeg` gets colors from the
  satellite palette the models learned.
- Outputs: `<out>` (side-by-side figure) and `<out stem>_raw.png`.

**Dashboard**: `python app.py` → http://127.0.0.1:7860 (or `$env:PORT`). *Colorize* has a
model dropdown per input type; *Results* shows every table above plus per-terrain and
training charts.

**Scripted demo**: `.\run_demo.ps1` (`-Fast` skips the evaluation step).

### Sharpness

`python _sharpness.py` measures variance of the Laplacian (higher = sharper) on the first
agri and urban tiles:

| image | agri | urban |
|---|---|---|
| ground truth (optical) | 398 | 2744 |
| raw SAR input | 2174 | 1167 (mostly speckle) |
| Model 2 output (full-res DnCNN luminance) | 225 | 100 |
| Model 2 output + unsharp mask | 1070 | 473 |
| Model 3 output | 3 | 2 |

Model 2's output inherits the DnCNN luminance's detail (composing at full resolution tripled
it vs the original 224-px pipeline, 72 → 225 on agri). **Model 3's output is very smooth**:
it predicts brightness itself from radar and gets the large-scale layout and colors right
but no fine texture — the main open problem (§7).

`--sharp` (dashboard: *Sharpen output*) applies a cosmetic unsharp mask, disclosed in the output.

---

## 4. Training

All trainers use the **canonical split** in [`splits.py`](splits.py): a stable hash of each
file name assigns it to train (80 %), val (10 %) or test (10 %), identically on every machine
and for every model. Each run writes `split.json` next to its checkpoint; `evaluate.py` uses it
to score only unseen pairs.

```powershell
python Train_model3.py            # SAR -> L + color; every improvement on (GPU recommended)
python Train_model1.py --loss bins --train_adapters --imagenet_norm --augment --out model1b_run
python Train_model1.py            # original Model 1 recipe
python train_dncnn.py; python Train_model2.py    # original Model 2 pipeline
python Train_model3.py --n_images 32 --n_val 16 --epochs 1 --batch_size 8 --out _smoke   # smoke test
```

All logic is in [`train_colorizer.py`](train_colorizer.py) (shared by Models 1–3) and
[`pipeline.py`](pipeline.py) (inputs, color bins, losses, decoding). Options: `--loss mse|l1|bins`,
`--predict_L`, `--train_adapters`, `--imagenet_norm`, `--augment` (each has a `--no_` form),
`--n_images`, `--n_val`, `--epochs`, `--batch_size`, `--lr`, `--scheduler`, `--bin_grid`, `--T`.
The best epoch is picked by validation a/b RMSE (+ L RMSE for Model 3). Multiple GPUs are used
automatically (`DataParallel`).

### Training on Kaggle (how Models 1b and 3 were trained)

[`kaggle/`](kaggle) runs everything as a private Kaggle GPU kernel (2× Tesla T4):

```powershell
python kaggle/prepare_code.py --user <kaggle-user> --out kcode      # flatten the repo
cd kcode/..; kaggle datasets create -p kcode                         # private dataset (once)
#   later: kaggle datasets version -p kcode -m "update"
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4            # edit the id in kernel-metadata.json
kaggle kernels output <kaggle-user>/colorsar-train -p kaggle_out    # when it finishes
```

`run_kaggle.py` trains Model 3 (30 epochs, ~97 min) and Model 1b (20 epochs, ~58 min), then
evaluates both. Training logs are kept in `model3_run/logs/` and `model1b_run/logs/`.
Run `kaggle datasets create`/`version` with a *relative* path (the CLI mishandles `C:` paths).

[`ColorSAR_Kaggle_Training.ipynb`](ColorSAR_Kaggle_Training.ipynb) is the notebook route for
Model 1 (clones this repo and runs `Train_model1.py`).

**About the DnCNN.** It keeps the original architecture (depth 17, residual `x − net(x)`), but
its task is translating SAR brightness into *optical* brightness — a different sensor — not
removing noise from one image. Model 3 drops it and learns that translation end to end.

---

## 5. Evaluation

```powershell
python evaluate.py --model 3 --per_terrain --out model3_run/metrics.json
python evaluate.py --model 1 --checkpoint model1b_run/model_1.pth --per_terrain
python evaluate.py --model 2 --intersect model3_run/split.json     # same pairs as Model 3's test
python evaluate.py --model 2 --n_images 200                        # quick check
python evaluate.py --model 2 --split all                           # all pairs (includes training data)
python train_dncnn.py --eval_only                                  # DnCNN stage
```

Printed rows: model, no-color baseline, mean-color baseline, oracle (true a/b on the same L —
a Lab round trip for Model 1), a/b RMSE for each, and the brightness error. `--intersect`
restricts scoring to pairs that another checkpoint also never trained on. Evaluation runs at
224×224, the training resolution.

---

## 6. Checkpoints and portability

- **Format 2** (Models 1b and 3, and anything `train_colorizer.py` writes): decoder +
  adapters + metadata (input type, normalization, head, color bins, training settings) in one
  file. Fully portable: Model 3 trained on Kaggle (torch 2.x CUDA) reproduces its scores on a
  CPU laptop.
- **Format 1** (the original `model_1.pth` / `model_2.pth`): decoder only. The adapters are
  rebuilt from `ENCODER_FROZEN_SEED = 42`, and `load_colorizer` **refuses** to load unless they
  match fingerprint `788e0728e71f`. This guard fired on Kaggle, whose newer PyTorch draws
  different random weights from the same seed (`1da5df7e60c8`) — exactly the silent failure
  the original project was exposed to. Use the pinned torch version for these two checkpoints.

---

## 7. Known limitations

- **Model 3 is blurry.** Its brightness comes from radar, which lacks optical detail, at
  224×224. Next steps: an adversarial or perceptual loss for texture, higher-resolution
  output, or fusing the predicted L with SAR edges.
- **Brightness still dominates Model 3's error** (L RMSE ≈ 15.8 Lab units).
- **The original Models 1 and 2 are kept for comparison**; their checkpoints depend on the
  pinned PyTorch version (see §6).
- **Domain**: trained on 256×256 Sentinel tiles only.

---

## 8. Files

```
├── model1_run/ model1b_run/ model2_run/ model3_run/
│       checkpoint(s), split.json, history.json, metrics*.json, loss_curve.png,
│       sample_predictions.png (+ logs/ for the Kaggle runs, dncnn files in model2_run)
├── model_architecture.py   encoder, decoder, checkpoint save/load (format 1 + 2)
├── pipeline.py             model inputs, color bins, rebalanced loss, annealed mean, decoding
├── color_utils.py          pairing, dataset, Lab conversions, SAR -> L stage
├── splits.py               canonical split, split.json, held-out selection, legacy splits
├── dncnn.py, train_dncnn.py               DnCNN (Model 2 front end)
├── train_colorizer.py                     trainer for Models 1-3
├── Train_model1.py, Train_model2.py, Train_model3.py   per-model defaults
├── kaggle/                 Kaggle GPU kernel (run_kaggle.py, kernel-metadata.json, prepare_code.py)
├── ColorSAR_Kaggle_Training.ipynb         notebook driver for Model 1
├── evaluate.py             held-out PSNR / SSIM / a/b + L RMSE with baselines
├── colorize.py             inference CLI (the one inference path)
├── colorize_image.py       alias for `colorize.py --mode gray`
├── app.py                  Gradio dashboard
├── run_demo.ps1            scripted demo
├── make_figures.py         regenerate a run's figures from its checkpoint
├── _sharpness.py           sharpness audit
├── visual_test.py, data_pipeline.py, count_pairs.py   sanity checks
├── requirements.txt        pinned dependencies
└── sample_input.jpeg       an everyday photo (out-of-domain example)
```
