"""
ColorSAR - model inputs, output decoding, color bins and losses

One place that knows how each kind of checkpoint turns a batch into encoder input and
how its raw decoder output becomes (L, a/b). Training, evaluation and inference all
call these functions, so they cannot disagree.

Inputs (meta["input"])
    "optical_L"   Model 1: L of the optical (s2) image
    "sar_dncnn_L" Model 2: L of the SAR (s1) tile after the DnCNN
    "sar3"        Model 3: the SAR tile itself as 3 channels - raw, Gaussian-smoothed
                  (sigma 1) and (sigma 2.5), i.e. the image at three despeckling levels -
                  ImageNet-normalized. No DnCNN: the network learns SAR -> optical itself.

Outputs (meta["ab_head"], meta["predict_L"])
    "tanh"  original head: normalized a/b directly
    "reg"   a/b regression (tanh applied here)
    "bins"  logits over Q color bins (Zhang et al. 2016, "Colorful Image Colorization"):
            trained with class-rebalanced cross-entropy on soft-encoded targets, decoded
            with the annealed mean (temperature T). Avoids the desaturated "average color"
            that MSE regression converges to when the color is uncertain.
    predict_L: the first channel is the L channel (tanh) - Model 3 predicts brightness too,
            instead of inheriting the DnCNN's (which capped Model 2 at ~14.6 dB PSNR).
"""
import math

import torch
import torch.nn.functional as F

from color_utils import SIZE, ab_to_lab_units, resize_t, rgb_to_L, sar_to_L
from model_architecture import _IMAGENET_MEAN, _IMAGENET_STD, encoder_input

INPUT_OF_MODEL = {1: "optical_L", 2: "sar_dncnn_L", 3: "sar3"}


# --------------------------------------------------------------------------- inputs
def _gauss(x, sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def sar3_input(gray, size=SIZE):
    """SAR gray (N,1,h,w) in [0,1] -> ImageNet-normalized (N,3,size,size)."""
    g = resize_t(gray, size)
    x = torch.cat([g, _gauss(g, 1.0), _gauss(g, 2.5)], dim=1)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def model_input(meta, gray=None, rgb=None, dncnn=None):
    """-> (encoder input (N,3,224,224), given L (N,1,224,224) or None for sar3).

    gray: SAR (N,1,h,w) in [0,1] at native resolution; rgb: optical (N,3,224,224) in [0,1].
    """
    kind = meta["input"]
    if kind == "optical_L":
        L = rgb_to_L(rgb)
    elif kind == "sar_dncnn_L":
        L, _ = sar_to_L(gray, dncnn, meta.get("dncnn_res", "native"))
    elif kind == "sar3":
        return sar3_input(gray), None
    else:
        raise ValueError(f"unknown input kind {kind!r}")
    return encoder_input(L, meta.get("input_norm", "none")), L


def forward_raw(encoder, decoder, enc_in):
    f56, f28, f14, f7 = encoder(enc_in)
    return decoder(f7, f14, f28, f56)


# --------------------------------------------------------------------------- bins
def lab_to_norm_ab(ab_lab):
    return (ab_lab + 128) * 2 / 255 - 1


def compute_bins(ab_norm, grid=10.0, min_count=1):
    """Color bins from training a/b. ab_norm: (M,2) normalized samples.

    Returns (centers (Q,2) in Lab units, prior (Q,) probabilities): the cells of a
    `grid`-unit lattice over a/b that actually occur in the data.
    """
    ab = ab_to_lab_units(ab_norm)
    idx = torch.round(ab / grid).long()
    keys, counts = torch.unique(idx, dim=0, return_counts=True)
    keep = counts >= min_count
    centers = keys[keep].float() * grid
    prior = counts[keep].float()
    return centers, prior / prior.sum()


def class_weights(centers, prior, lam=0.5, sigma=5.0):
    """Zhang et al. rebalancing: w ~ 1 / ((1-lam) * smoothed prior + lam / Q), E_p[w] = 1."""
    d2 = torch.cdist(centers, centers).pow(2)
    k = torch.exp(-d2 / (2 * sigma ** 2))
    smooth = (k @ prior)
    smooth = smooth / smooth.sum()
    w = 1.0 / ((1 - lam) * smooth + lam / len(prior))
    return w / (smooth * w).sum()


def bin_loss(logits, ab_norm, centers, weights, stride=4, k=5, sigma=5.0):
    """Class-rebalanced cross-entropy against soft-encoded a/b targets.

    Evaluated on every `stride`-th pixel (224 -> 56 per side) to keep memory small.
    """
    lg = logits[:, :, ::stride, ::stride].flatten(2).transpose(1, 2)           # (N,P,Q)
    ab = ab_to_lab_units(ab_norm[:, :, ::stride, ::stride]).flatten(2).transpose(1, 2)  # (N,P,2)
    d2 = ((ab.unsqueeze(2) - centers.view(1, 1, -1, 2)) ** 2).sum(-1)          # (N,P,Q)
    kk = min(k, centers.shape[0])
    dk, ik = d2.topk(kk, dim=-1, largest=False)
    soft = torch.softmax(-dk / (2 * sigma ** 2), dim=-1)
    target = torch.zeros_like(lg).scatter_(-1, ik, soft)
    ce = -(target * F.log_softmax(lg, dim=-1)).sum(-1)                          # (N,P)
    return (ce * weights[ik[..., 0]]).mean()


def annealed_mean(logits, centers, T=0.38):
    """Bin logits (N,Q,H,W) -> normalized a/b (N,2,H,W). T<1 sharpens toward the mode."""
    p = torch.softmax(logits / T, dim=1)
    ab = torch.einsum("nqhw,qc->nchw", p, centers.to(p.dtype))
    return lab_to_norm_ab(ab)


# --------------------------------------------------------------------------- outputs
def decode_output(raw, meta, centers=None):
    """Raw decoder output -> (predicted L (N,1,H,W) or None, normalized a/b (N,2,H,W))."""
    head = meta.get("ab_head", "tanh")
    if head == "tanh" and not meta.get("predict_L"):
        return None, raw
    L = torch.tanh(raw[:, :1]) if meta.get("predict_L") else None
    rest = raw[:, 1:] if L is not None else raw
    if head == "reg":
        return L, torch.tanh(rest)
    if centers is None:
        centers = torch.tensor(meta["bins"], dtype=raw.dtype, device=raw.device)
    return L, annealed_mean(rest, centers, meta.get("T", 0.38))


@torch.no_grad()
def predict(encoder, decoder, meta, gray=None, rgb=None, dncnn=None):
    """Full inference on a batch -> (L used for the output (N,1,224,224), a/b normalized)."""
    enc_in, L_given = model_input(meta, gray=gray, rgb=rgb, dncnn=dncnn)
    L_pred, ab = decode_output(forward_raw(encoder, decoder, enc_in), meta)
    return (L_pred if L_pred is not None else L_given), ab
