"""
ColorSAR - train Model 2 (DnCNN-processed SAR L -> optical a/b).

Defaults reproduce the gray_model.ipynb recipe behind model2_run/model_2.pth:
8000 training pairs, 15 epochs, batch 32, Adam 1e-3. All logic lives in
train_colorizer.py; see it for every option.

    python train_dncnn.py              # once, writes model2_run/dncnn.pth
    python Train_model2.py
    python Train_model2.py --no_denoise                       # ablation: raw SAR L
    python Train_model2.py --n_images 32 --epochs 1 --out _smoke   # smoke test
"""
from train_colorizer import main

if __name__ == "__main__":
    main(2)
