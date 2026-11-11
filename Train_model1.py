"""
ColorSAR - train Model 1 (optical L -> a/b).

Defaults reproduce the run behind model1_run/model_1.pth (Kaggle, 2x T4):
all training pairs, 20 epochs, batch 64, Adam 5e-4 with cosine decay to 1e-5.
Multiple GPUs are used automatically (DataParallel). All logic lives in
train_colorizer.py; see it for every option.

    python Train_model1.py                              # full recipe
    python Train_model1.py --n_images 2000 --epochs 3 --batch_size 8   # quick laptop run
"""
from train_colorizer import main

if __name__ == "__main__":
    main(1)
