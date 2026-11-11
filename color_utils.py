"""
ColorSAR - Shared helpers (dataset pairing, Lab conversions, the SAR -> L stage)

Used by every training, evaluation and inference script, so they all agree on
exactly how s1/s2 pairs are matched, how images are resized and how Lab channels
are normalized. Nothing in this file should be re-implemented elsewhere.

Pairing rule (verified on v_2): the 4 terrains each hold s1/ (SAR, 8-bit
grayscale) and s2/ (optical, 8-bit RGB) with identical filenames except the
`_s1_` / `_s2_` token -> 16,000 exact pairs.
"""
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from torch.utils.data import Dataset

SIZE = 224  # the colorizers work on 224x224 inputs


# --------------------------------------------------------------------------- pairing
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


def is_sentinel1_path(path):
    """True for files that are Sentinel-1 SAR tiles by name/location (`_s1_` or an s1/ dir)."""
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    return "_s1_" in os.path.basename(path) or (len(parts) > 1 and parts[-2] == "s1")


# --------------------------------------------------------------------------- dataset
class PairDataset(Dataset):
    """One s1/s2 pair -> (sar_gray, rgb, ab, terrain).

    sar_gray : (1,h,w) float in [0,1] at the file's NATIVE resolution (256 for v_2),
               so the DnCNN can run on the scale it was trained on (see sar_to_L).
    rgb      : (3,224,224) float in [0,1] - optical ground truth
    ab       : (2,224,224) float in [-1,1] - optical a/b target
    """

    def __init__(self, pairs, size=SIZE, load_sar=True):
        self.pairs = pairs
        self.size = size
        self.load_sar = load_sar

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sar_path, opt_path, terrain = self.pairs[idx]
        if self.load_sar:
            gray = np.array(Image.open(sar_path).convert("L"), dtype=np.float32) / 255.0
        else:
            gray = np.zeros((1, 1), dtype=np.float32)
        rgb = np.array(Image.open(opt_path).convert("RGB"))
        rgb = cv2.resize(rgb, (self.size, self.size))
        _, ab = rgb_to_lab(rgb)
        rgb_t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) / 255.0
        return torch.from_numpy(gray).unsqueeze(0), rgb_t, torch.from_numpy(ab), terrain


def resize_t(x, size=SIZE):
    """Bilinear resize of an (N,C,H,W) tensor to size x size (no-op if already there)."""
    if x.shape[-2:] == (size, size):
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def sar_to_L(gray, dncnn, dncnn_res="native", size=SIZE, need_native=False):
    """SAR gray (N,1,h,w) in [0,1] -> (L for the colorizer at size x size, native gray).

    dncnn_res="native": DnCNN runs on the full-resolution tile (the scale it was
                        trained on), then the result is resized - used by new runs.
    dncnn_res="224"   : resize first, then DnCNN - how the shipped model_2.pth was
                        trained, kept so that checkpoint sees exactly its training input.
    The second value is the (denoised) gray at the input's native resolution, used to
    compose full-resolution output; it is None unless need_native=True.
    """
    if dncnn_res not in ("native", "224"):
        raise ValueError(f"unknown dncnn_res: {dncnn_res!r}")
    if dncnn is None:
        return gray_to_L(resize_t(gray, size)), gray
    native = None
    with torch.no_grad():
        if dncnn_res == "native" or need_native:
            native = dncnn(gray).clamp(0.0, 1.0)
        if dncnn_res == "native":
            small = resize_t(native, size)
        else:
            small = dncnn(resize_t(gray, size)).clamp(0.0, 1.0)
    return gray_to_L(small), native


# --------------------------------------------------------------------------- Lab
def rgb_to_lab(img):
    """RGB uint8 (H,W,3) -> (L, ab) float32 CHW, both scaled to [-1, 1].

    Same normalization as the original notebook. skimage's rgb2lab divides uint8
    input by 255 itself.
    """
    lab = rgb2lab(img).astype("float32")
    lab = np.transpose(lab, (2, 0, 1))
    L = 2 * lab[[0], ...] / 100 - 1
    ab = 2 * (lab[[1, 2], ...] + 128) / 255 - 1
    return L, ab


# Normalized value that corresponds to a = b = 0 (no color).
AB_NEUTRAL = 2 * 128 / 255 - 1


def ab_to_lab_units(ab):
    """Normalized ab tensor/array in [-1,1] -> Lab a*/b* units."""
    return (ab + 1) * 0.5 * 255 - 128


# sRGB -> XYZ (D65) and D65 white point: the constants skimage.color uses internally,
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
    """Normalized (N,1,H,W)/(N,2,H,W) tensors -> (N,H,W,3) float RGB in [0,1].

    If ab has a lower resolution than L it is bilinearly upsampled first, so a
    224x224 color prediction can be composed with a full-resolution L.
    """
    if ab.shape[-2:] != L.shape[-2:]:
        ab = F.interpolate(ab, size=L.shape[-2:], mode="bilinear", align_corners=False)
    L_u = (L.detach().cpu().numpy() + 1) * 0.5 * 100
    ab_u = ab_to_lab_units(ab.detach().cpu().numpy())
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
