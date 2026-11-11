"""
ColorSAR - Data pipeline sanity check (Lab conversion + patching)

The conversion itself lives in color_utils.py (shared by every script); this file
only demonstrates it on one image and keeps the patch helper from the original
notebook (original_model.ipynb, cells 9-11; https://github.com/nishant9083/ColorSAR).

    python data_pipeline.py v_2/agri/s2/<file>.png
"""
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from color_utils import SIZE, lab_to_rgb, rgb_to_lab


def create_patches(img, patch_size=SIZE):
    """Splits a (C,H,W) image into non-overlapping patch_size x patch_size patches."""
    patches = []
    h, w = img.shape[1:]
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img[:, i:i + patch_size, j:j + patch_size]
            if patch.shape[1:] == (patch_size, patch_size):
                patches.append(patch)
    return patches


def load_and_preprocess(image_path, size=SIZE):
    """Loads one optical image, resizes it, returns normalized (L, ab) numpy arrays."""
    color_img = cv2.resize(np.array(Image.open(image_path).convert("RGB")), (size, size))
    return rgb_to_lab(color_img)


if __name__ == "__main__":
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    assert test_image, "Pass a test image path"
    print(f"Testing pipeline on: {test_image}")

    L, ab = load_and_preprocess(test_image)
    print(f"L channel shape:  {L.shape}   range: [{L.min():.3f}, {L.max():.3f}]")
    print(f"ab channels shape: {ab.shape}   range: [{ab.min():.3f}, {ab.max():.3f}]")

    rgb = lab_to_rgb(torch.from_numpy(L)[None], torch.from_numpy(ab)[None])[0]
    orig = cv2.resize(np.array(Image.open(test_image).convert("RGB")), (SIZE, SIZE)) / 255.0
    print(f"Reconstructed RGB shape: {rgb.shape}, max abs round-trip error: "
          f"{np.abs(rgb - orig).max():.4f}")

    patches = create_patches(np.random.rand(3, 500, 500).astype("float32"))
    print(f"Patches extracted from 500x500 image: {len(patches)} (expected 4), "
          f"each {patches[0].shape}")
    print("\nALL PIPELINE STEPS RAN SUCCESSFULLY.")
