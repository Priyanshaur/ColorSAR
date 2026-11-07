"""
ColorSAR - Evaluation: PSNR / SSIM for a trained colorizer

The original repo quotes (ColorSAR_Project_Report.pdf / Readme.md):
    Model 1  PSNR 29.9230   SSIM 0.9400
    Model 2  PSNR 14.4466   SSIM 0.2314

This script measures the same kind of numbers on v_2:
    Model 1 : L taken from the optical (s2) image  -> upper-bound / clean input
    Model 2 : L taken from the SAR (s1) image (passed through DnCNN first)
and reports the reconstruction RGB (L + predicted ab) against the s2 ground truth.

It also prints an "oracle" row (true ab with the same L) = the ceiling imposed by
the luminance you feed in, which separates "bad colors" from "bad luminance".

Usage:
    python evaluate.py --model 1 --checkpoint model1_run/model_1.pth
    python evaluate.py --model 2 --checkpoint model2_run/model_2.pth --per_terrain
    python evaluate.py --model 1 --checkpoint model1_run/model_1.pth --n_images 500
"""
import argparse
import json

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

from color_utils import (find_pairs, gray_to_L, lab_to_rgb, load_decoder,
                         rgb_to_L, rgb_to_lab)
from dncnn import load_dncnn
from model_architecture import EnsembleEncoder, encoder_fingerprint


class EvalDataset(Dataset):
    """Yields (gray from s1, rgb from s2, ab from s2, terrain) at 224x224."""

    def __init__(self, pairs, size=224):
        self.pairs = pairs
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sar_path, opt_path, terrain = self.pairs[idx]

        gray = np.array(Image.open(sar_path).convert("L"), dtype=np.float32) / 255.0
        gray = cv2.resize(gray, (self.size, self.size))

        rgb = np.array(Image.open(opt_path).convert("RGB"))
        rgb = cv2.resize(rgb, (self.size, self.size))
        _, ab = rgb_to_lab(rgb)

        gray_t = torch.from_numpy(gray).unsqueeze(0)
        rgb_t = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1) / 255.0
        return gray_t, rgb_t, torch.from_numpy(ab), terrain


@torch.no_grad()
def run(model_input, encoder, decoder):
    f56, f28, f14, f7 = encoder(model_input)
    return decoder(f7, f14, f28, f56)


def psnr_ssim(pred, gt):
    p = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    s = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
    return p, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, choices=[1, 2], default=1)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--n_images", type=int, default=0, help="0 = evaluate on every pair")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dncnn", default="model2_run/dncnn.pth")
    ap.add_argument("--no_denoise", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--per_terrain", action="store_true")
    ap.add_argument("--out", default=None, help="optional path for a JSON metrics file")
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = ("model1_run/model_1.pth" if args.model == 1
                           else "model2_run/model_2.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model {args.model}  |  checkpoint: {args.checkpoint}")

    dncnn = None
    if args.model == 2 and not args.no_denoise:
        dncnn = load_dncnn(args.dncnn, device)
        print(f"Input luminance: SAR -> DnCNN ({args.dncnn}) -> L")
    else:
        print(f"Input luminance: {'optical (s2) L' if args.model == 1 else 'raw SAR (s1) L'}")

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    if args.n_images > 0:
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(pairs))[:args.n_images]
        pairs = [pairs[i] for i in idx]
    print(f"Evaluating on {len(pairs)} pairs ...")

    loader = DataLoader(EvalDataset(pairs), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    encoder = EnsembleEncoder(pretrained=True).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Encoder frozen-layer fingerprint: {encoder_fingerprint(encoder)} "
          "(must match the value printed when the checkpoint was trained)")
    decoder = load_decoder(args.checkpoint, device)

    stats = {}          # terrain -> [psnr_sum, ssim_sum, oracle_psnr_sum, n]
    tot_p = tot_s = tot_op = 0.0
    n = 0

    for gray, rgb_gt, ab_gt, terrain in loader:
        gray, rgb_gt, ab_gt = gray.to(device), rgb_gt.to(device), ab_gt.to(device)

        if args.model == 1:
            L = rgb_to_L(rgb_gt)                      # clean-input (Model 1) setup
        else:
            if dncnn is not None:
                gray = dncnn(gray).clamp(0.0, 1.0)
            L = gray_to_L(gray)                              # SAR-derived (Model 2) setup

        pred_ab = run(L.repeat(1, 3, 1, 1), encoder, decoder)

        pred_rgb = lab_to_rgb(L, pred_ab)                   # (N,H,W,3) numpy
        oracle_rgb = lab_to_rgb(L, ab_gt)
        gt = rgb_gt.permute(0, 2, 3, 1).cpu().numpy()       # (N,H,W,3) numpy
        pred = pred_rgb
        orc = oracle_rgb

        for i in range(gt.shape[0]):
            p, s = psnr_ssim(pred[i], gt[i])
            op, _ = psnr_ssim(orc[i], gt[i])
            t = terrain[i]
            rec = stats.setdefault(t, [0.0, 0.0, 0.0, 0])
            rec[0] += p; rec[1] += s; rec[2] += op; rec[3] += 1
            tot_p += p; tot_s += s; tot_op += op; n += 1

    print("\n" + "=" * 62)
    print(f"Model {args.model}  —  {n} images")
    print("=" * 62)
    print(f"  PSNR (predicted color vs ground truth) : {tot_p/n:.4f} dB")
    print(f"  SSIM (predicted color vs ground truth) : {tot_s/n:.4f}")
    print(f"  Oracle PSNR (true ab, same L)          : {tot_op/n:.4f} dB")
    if args.model == 1:
        print("  Upstream ColorSAR Model 1 reference    : PSNR 29.9230  SSIM 0.9400")
    else:
        print("  Upstream ColorSAR Model 2 reference    : PSNR 14.4466  SSIM 0.2314")

    if args.per_terrain and stats:
        print("\n  Per terrain:")
        print(f"  {'terrain':<14}{'n':>6}{'PSNR':>10}{'SSIM':>10}{'oracle PSNR':>14}")
        for t in sorted(stats):
            ps, ss, os_, cnt = stats[t]
            print(f"  {t:<14}{cnt:>6}{ps/cnt:>10.4f}{ss/cnt:>10.4f}{os_/cnt:>14.4f}")

    if args.out:
        result = {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "n_images": n,
            "psnr": float(tot_p / n),
            "ssim": float(tot_s / n),
            "oracle_psnr": float(tot_op / n),
            "per_terrain": {t: {"n": v[3], "psnr": float(v[0] / v[3]), "ssim": float(v[1] / v[3]),
                                "oracle_psnr": float(v[2] / v[3])} for t, v in stats.items()},
        }
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"\nSaved metrics to {args.out}")
