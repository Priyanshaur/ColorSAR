"""
ColorSAR - Shared helpers (dataset pairing + Lab conversions)

Used by train_dncnn.py, Train_model2.py and evaluate.py so all three agree
on exactly how s1/s2 pairs are matched and how Lab channels are normalized.

Pairing rule (verified on v_2): the 4 terrains each hold s1/ (SAR, 8-bit
grayscale) and s2/ (optical, 8-bit RGB) with identical filenames except the
`_s1_` / `_s2_` token -> 16,000 exact pairs.
"""
import os
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.color import rgb2lab, lab2rgb

from model_architecture import Decoder


def find_pairs(root_dir="v_2"):
    """Return [(sar_path, optical_path, terrain), ...] matched by filename."""
    pairs = []
    if not os.path.isdir(root_dir):
        return pairs

    for terrain in sorted(os.listdir(root_dir)):
        s1_dir = os.path.join(root_dir, terrain, "s1")
        s2_dir = os.path.join(root_dir, terrain, "s2")
        if not (os.path.isdir(s1_dir) and os.path.isdir(s2_dir)):
            continue

        optical = {f.replace("_s2_", "_X_"): os.path.join(s2_dir, f)
                   for f in os.listdir(s2_dir) if f.endswith(".png")}

        for f in sorted(os.listdir(s1_dir)):
            if not f.endswith(".png"):
                continue
            key = f.replace("_s1_", "_X_")
            if key in optical:
                pairs.append((os.path.join(s1_dir, f), optical[key], terrain))
    return pairs


def load_decoder(path, device):
    """Load decoder weights saved by Train_model1.py / Train_model2.py / Kaggle notebooks.

    Tolerant loader: accepts a plain decoder state_dict, a whole ColorizationModel
    (keys prefixed `decoder.`) and DataParallel checkpoints (keys prefixed `module.`).
    """
    assert os.path.exists(path), f"Checkpoint not found: {path}"
    obj = torch.load(path, map_location=device, weights_only=False)
    state = obj.state_dict() if isinstance(obj, nn.Module) else obj

    cleaned = {}
    for k, v in state.items():
        k2 = k[len("module."):] if k.startswith("module.") else k
        k2 = k2[len("decoder."):] if k2.startswith("decoder.") else k2
        cleaned[k2] = v
    decoder = Decoder()
    decoder.load_state_dict(cleaned)
    return decoder.to(device).eval()


def rgb_to_lab(img):
    """RGB uint8 (H,W,3) -> (L, ab) float32 CHW, both scaled to [-1, 1].

    Same normalization as the original notebook (and Train_model1.py).
    skimage's rgb2lab divides uint8 input by 255 itself.
    """
    lab = rgb2lab(img).astype("float32")
    lab = np.transpose(lab, (2, 0, 1))
    L = 2 * lab[[0], ...] / 100 - 1
    ab = 2 * (lab[[1, 2], ...] + 128) / 255 - 1
    return L, ab


# sRGB -> XYZ (D65) and D65 white point: the constants skimage.metrics uses internally,
# so the torch versions below agree with skimage.color.rgb2lab to float precision.
_RGB_TO_XYZ = ((0.4124564, 0.3575761, 0.1804375),
               (0.2126729, 0.7151522, 0.0721750),
               (0.0193339, 0.1191920, 0.9503041))
_WHITE_D65 = (0.95047, 1.00000, 1.08883)


def _srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).pow(2.4))


def _f_l(t):
    delta = 6.0 / 29.0
    return torch.where(t > delta ** 3,
                       t.clamp_min(1e-12).pow(1.0 / 3.0),
                       t / (3.0 * delta * delta) + 4.0 / 29.0)


def rgb_to_L(rgb):
    """RGB tensor (N,3,H,W) in [0,1] -> Lab L (N,1,H,W) in [-1,1]."""
    linear = _srgb_to_linear(rgb.clamp(0.0, 1.0))
    m = torch.tensor(_RGB_TO_XYZ, dtype=linear.dtype, device=linear.device)
    xyz = torch.einsum("ij,njhw->nihw", m, linear)
    y = (xyz[:, 1] / _WHITE_D65[1]).clamp_min(0.0)
    return (2.0 * (116.0 * _f_l(y) - 16.0) / 100.0 - 1.0).unsqueeze(1)


def gray_to_L(gray):
    """Grayscale tensor in [0,1] (any leading dims, HxW at the end) -> Lab L in [-1, 1].

    Mathematically identical to `rgb2lab(cv2.merge([g, g, g]))[..., 0]` (what
    gray_model.ipynb uses), but pure torch, so it also runs on the *denoised*
    gray batch produced by DnCNN on the GPU inside the training loop.
    """
    c = gray.clamp(0.0, 1.0)
    return 2.0 * (116.0 * _f_l(_srgb_to_linear(c)) - 16.0) / 100.0 - 1.0


def lab_to_rgb(L, ab):
    """Normalized (N,1,H,W)/(N,2,H,W) tensors -> (N,H,W,3) float RGB in [0,1]."""
    L_u = (L.detach().cpu().numpy() + 1) * 0.5 * 100
    ab_u = (ab.detach().cpu().numpy() + 1) * 0.5 * 255 - 128
    lab = np.concatenate([L_u, ab_u], axis=1)              # (N,3,H,W)
    lab = np.transpose(lab, (0, 2, 3, 1))                  # (N,H,W,3)
    with warnings.catch_warnings():
        # lab2rgb complains about the few pixels where the 255 (instead of 256)
        # ab scaling pushes Lab slightly out of gamut; we clip right after anyway.
        warnings.simplefilter("ignore")
        rgb = np.stack([lab2rgb(img) for img in lab])
    return np.clip(rgb, 0.0, 1.0)


def unsharp(img, amount=1.2, radius=3.0):
    """Cosmetic unsharp mask on a float RGB image in [0,1]: sharpen existing edges.

    Boosts what detail is already present (the model output's structure comes from
    the luminance, so this can only re-emphasize it - it cannot invent real detail).
    """
    x = img.astype(np.float32)
    blur = cv2.GaussianBlur(x, (0, 0), radius)
    return np.clip(x + amount * (x - blur), 0.0, 1.0)
