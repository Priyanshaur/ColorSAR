# Model 1 checkpoint & results

ColorSAR Model 1: luminance (L) of an optical Sentinel-2 tile -> its a/b color channels.

## Files

| file | what it is |
|---|---|
| `model_1.pth` | decoder weights (format 1: decoder-only state_dict, 31.3 MB). Needs encoder fingerprint `788e0728e71f` - `load_colorizer` checks this and refuses to load otherwise |
| `split.json` | the exact train/val pairs this checkpoint used (reconstructed from the notebook's recipe by `python splits.py --legacy`) |
| `history.json` | per-epoch train/val loss of the run that produced `model_1.pth` |
| `metrics.json` | held-out evaluation written by `evaluate.py` (model vs no-color / mean-color baselines, a/b RMSE, per terrain) |
| `loss_curve.png`, `sample_predictions.png` | regenerated from this checkpoint by `python make_figures.py --model 1` |
| `colorized_sample_test.png` | `python colorize.py --image sample_input.jpeg` - an everyday photo, i.e. **out of domain** for a model trained on satellite tiles |

## Training run

- Kaggle, 2x Tesla T4, `nn.DataParallel`, batch 64
- all 16,000 pairs: 14,400 train / 1,600 val (random_split, seed 42) - **no test set was held out**, so
  `evaluate.py` scores this checkpoint on its 1,600 validation pairs (which also chose the best epoch:
  a slightly optimistic estimate). Retraining with `Train_model1.py` uses the canonical 80/10/10 split instead.
- 20 epochs, Adam 5e-4, CosineAnnealingLR down to 1e-5, MSE on normalized a/b
- frozen ImageNet ResNet-50 (IMAGENET1K_V2) + DenseNet-121 (IMAGENET1K_V1); adapters seeded (42) and frozen

## Loss history (from `history.json`)

| epoch | train MSE | val MSE |
|:---:|:---:|:---:|
| 1 | 0.0335 | 0.0423 |
| 2 | 0.0174 | 0.0174 |
| 3 | 0.0129 | 0.0093 |
| 4 | 0.0140 | 0.0229 |
| 5 | 0.0147 | 0.0140 |
| 6 | 0.0174 | 0.0098 |
| 7 | 0.0124 | 0.0083 |
| 8 | 0.0117 | 0.0130 |
| 9 | 0.0091 | 0.0070 |
| 10 | 0.0098 | 0.0138 |
| 11 | 0.0094 | 0.0067 |
| 12 | 0.0074 | 0.0064 |
| 13 | 0.0073 | 0.0052 |
| 14 | 0.0063 | 0.0057 |
| 15 | 0.0061 | 0.0050 |
| 16 | 0.0057 | 0.0054 |
| 17 | 0.0057 | 0.0055 |
| 18 | 0.0055 | 0.0044  **best - this checkpoint** |
| 19 | 0.0053 | 0.0045 |
| 20 | 0.0055 | 0.0045 |

Validation loss is noisy in the first half (e.g. epochs 4, 8, 10) and settles once the learning rate decays.

## Loading

```python
from model_architecture import load_colorizer, predict_ab
encoder, decoder, meta = load_colorizer("model1_run/model_1.pth", device)
ab = predict_ab(encoder, decoder, L, meta["input_norm"])   # L: (N,1,224,224) in [-1,1]
```
or simply `python colorize.py --image <file>`.
