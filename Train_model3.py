"""
ColorSAR - train Model 3: SAR tile -> optical L + a/b directly (no DnCNN bottleneck).

Defaults: all canonical training pairs, 30 epochs, batch 64, Adam 5e-4 + cosine,
color-bin classification loss + L1 on L, trained adapters, ImageNet-normalized input,
rot90/flip augmentation. Meant for a GPU (Kaggle 2x T4: see kaggle/).

    python Train_model3.py
    python Train_model3.py --n_images 32 --n_val 16 --epochs 1 --batch_size 8 --out _smoke
"""
from train_colorizer import main

if __name__ == "__main__":
    main(3)
