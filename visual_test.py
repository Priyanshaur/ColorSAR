"""
ColorSAR - Visual Sanity Check (untrained model)

Runs a REAL image from the dataset through the encoder-decoder with RANDOM weights
and saves the "colorization". It is expected to look like colored noise: the point
is only that load -> Lab -> encoder -> decoder -> reconstruct -> save runs end to end.
Uses the same model and preprocessing code as training (no private copies).

Usage:
    python visual_test.py "v_2/agri/s2/some_image.png"
"""
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from color_utils import SIZE, lab_to_rgb, rgb_to_lab
from model_architecture import ColorizationModel, Decoder, EnsembleEncoder, encoder_input

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


if __name__ == "__main__":
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    assert test_image, "Pass an image path, e.g. python visual_test.py v_2/agri/s2/xxx.png"

    print(f"Loading real image: {test_image}")
    rgb = cv2.resize(np.array(Image.open(test_image).convert("RGB")), (SIZE, SIZE))
    L, _ = rgb_to_lab(rgb)
    L = torch.from_numpy(L).unsqueeze(0)                          # (1,1,224,224)

    print("Building model (untrained / random weights)...")
    model = ColorizationModel(EnsembleEncoder(pretrained=False), Decoder()).eval()
    with torch.no_grad():
        pred_ab = model(encoder_input(L))                         # (1,2,224,224) in [-1,1]
    rgb_pred = lab_to_rgb(L, pred_ab)[0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(rgb); axes[0].set_title("Real Optical Image (ground truth)")
    axes[1].imshow(rgb_pred); axes[1].set_title("Model Output (UNTRAINED)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("visual_test_output.png", dpi=150)
    print("\nSaved: visual_test_output.png  (expected to look like noise - random weights)")
