"""
ColorSAR - Evaluation: PSNR / SSIM / color error for a trained colorizer

    Model 1 : L taken from the optical (s2) image   -> clean-input setting
    Model 2 : L taken from the SAR (s1) image, passed through the DnCNN first
    Model 3 : L and ab both predicted from the SAR (s1) tile
RGB is reconstructed from (L + predicted ab) and scored against the s2 image. The
"L RMSE" column is the brightness error vs the optical L (0 for Model 1 by design).

What gets scored (default --split heldout)
    Only pairs the checkpoint never trained on, read from the split.json written
    next to it (and, for Model 2, the DnCNN's split too). See splits.heldout.
    --split all scores all 16,000 pairs, which includes training data.

What gets printed - always read the rows together
    model         the checkpoint's colors
    no color      the same L with a = b = 0 (a grayscale image). PSNR/SSIM on RGB are
                  dominated by luminance; the gap between "model" and "no color" is
                  what the colorizer actually contributes.
    mean color    the same L with one constant a/b everywhere: the average color of
                  500 of the checkpoint's TRAINING images. A colorizer has to beat this
                  trivial "paint everything the average tint" predictor to be useful.
    oracle        the same L with the TRUE ab. For Model 1 this is just a Lab round
                  trip (~128 dB, not a meaningful ceiling). For Model 2 it is the
                  score a perfect colorizer would get given that luminance.
    ab RMSE       error of the a/b channels in Lab units (lower = better). Independent
                  of luminance, so it is the direct measure of color quality.

The upstream ColorSAR numbers (Model 1 29.9230 / 0.9400, Model 2 14.4466 / 0.2314)
come from a different data selection and split; they are context, not a like-for-like
comparison.

Usage:
    python evaluate.py --model 1 --per_terrain --out model1_run/metrics.json
    python evaluate.py --model 2 --per_terrain --out model2_run/metrics.json
    python evaluate.py --model 2 --n_images 200          # quick check on 200 held-out pairs
    python evaluate.py --model 3 --intersect model2_run/split.json   # same pairs as Model 2's test
"""
import argparse
import json
import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from color_utils import AB_NEUTRAL, PairDataset, ab_to_lab_units, find_pairs, lab_to_rgb, rgb_to_L
from dncnn import load_dncnn
from model_architecture import load_colorizer
from pipeline import INPUT_OF_MODEL, predict
from splits import heldout, pair_key, read_split, subsample

UPSTREAM = {1: (29.9230, 0.9400), 2: (14.4466, 0.2314), 3: (14.4466, 0.2314)}
ROWS = ("model", "no_color", "mean_color", "oracle")
LABELS = {"model": "model (predicted color)", "no_color": "no-color baseline (a=b=0)",
          "mean_color": "mean-color baseline (train avg)", "oracle": "oracle (true ab, same L)"}


def psnr_ssim(pred, gt):
    p = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    s = structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
    return p, s


def ab_rmse(ab_pred, ab_gt):
    """Per-image RMSE of a/b in Lab units. Tensors (N,2,H,W), normalized."""
    d = ab_to_lab_units(ab_pred) - ab_to_lab_units(ab_gt)
    return d.pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy()


def summarize(recs):
    """recs: per-image [psnr, ssim] for each row, ab RMSE for each row, then L RMSE."""
    a = np.array(recs)
    k = 2 * len(ROWS)
    out = {r: {"psnr": float(a[:, 2 * i].mean()), "ssim": float(a[:, 2 * i + 1].mean()),
               "ab_rmse": float(a[:, k + i].mean())} for i, r in enumerate(ROWS)}
    out["L_rmse"] = float(a[:, -1].mean())
    out["n"] = len(a)
    return out


def mean_train_ab(all_pairs, split_path, n=500):
    """Average normalized (a, b) over up to n of the checkpoint's training images."""
    train = set(read_split(split_path)["train"]) if os.path.exists(split_path) else None
    pool = [p for p in all_pairs if train is None or pair_key(p) in train]
    ds = PairDataset(subsample(pool, n, seed=0), load_sar=False)
    return torch.stack([ds[i][2].mean(dim=(1, 2)) for i in range(len(ds))]).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--split", choices=["heldout", "all"], default="heldout")
    ap.add_argument("--n_images", type=int, default=0, help="0 = every pair of the chosen split")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dncnn", default="model2_run/dncnn.pth")
    ap.add_argument("--no_denoise", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--per_terrain", action="store_true")
    ap.add_argument("--intersect", default=None, metavar="SPLIT_JSON",
                    help="also require pairs to be held out by this other checkpoint's split, "
                         "so two models are scored on exactly the same unseen pairs")
    ap.add_argument("--out", default=None, help="optional path for a JSON metrics file")
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = f"model{args.model}_run/model_{args.model}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model {args.model}  |  checkpoint: {args.checkpoint}")
    encoder, decoder, meta = load_colorizer(args.checkpoint, device)
    meta.setdefault("input", INPUT_OF_MODEL[args.model])
    print(f"Checkpoint format {meta.get('format', 2)}  |  encoder fingerprint {meta['fingerprint']}"
          f"  |  input_norm {meta['input_norm']}")

    dncnn = None
    if args.model == 2 and not args.no_denoise:
        dncnn = load_dncnn(args.dncnn, device)
        print(f"Input luminance: SAR -> DnCNN ({args.dncnn}, run at {meta['dncnn_res']} res) -> L")
    else:
        print("Input: " + {1: "optical (s2) L", 2: "raw SAR (s1) L",
                           3: "SAR (s1) tile; L and ab predicted"}[args.model])

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    split_path = os.path.join(os.path.dirname(args.checkpoint) or ".", "split.json")
    if args.split == "heldout":
        if not os.path.exists(split_path):
            raise SystemExit(f"{split_path} not found - cannot tell which pairs this checkpoint "
                             "trained on. Use --split all (includes training data) or add the file.")
        pairs_eval, desc = heldout(pairs, split_path)
        if args.intersect:
            other = {pair_key(p) for p in heldout(pairs, args.intersect)[0]}
            pairs_eval = [p for p in pairs_eval if pair_key(p) in other]
            desc += f", intersected with pairs held out by {args.intersect} -> {len(pairs_eval)}"
    else:
        pairs_eval, desc = pairs, f"ALL {len(pairs)} pairs - includes training data, optimistic"
    pairs_eval = subsample(pairs_eval, args.n_images, seed=42)
    print(f"Evaluating on {len(pairs_eval)} pairs: {desc}")
    mean_ab = mean_train_ab(pairs, split_path).to(device)

    loader = DataLoader(PairDataset(pairs_eval, load_sar=args.model != 1),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    stats, allrecs = {}, []
    with torch.no_grad():
        for gray, rgb_gt, ab_gt, terrain in loader:
            rgb_gt, ab_gt = rgb_gt.to(device), ab_gt.to(device)
            L, pred_ab = predict(encoder, decoder, meta, gray=gray.to(device), rgb=rgb_gt, dncnn=dncnn)
            L_rmse = ((L - rgb_to_L(rgb_gt)) * 50).pow(2).mean(dim=(1, 2, 3)).sqrt().cpu().numpy()
            neutral = torch.full_like(ab_gt, AB_NEUTRAL)
            const = mean_ab.view(1, 2, 1, 1).expand_as(ab_gt)

            gt = rgb_gt.permute(0, 2, 3, 1).cpu().numpy()
            abs_ = [pred_ab, neutral, const, ab_gt]                  # same order as ROWS
            imgs = [lab_to_rgb(L, x) for x in abs_]
            errs = [ab_rmse(x, ab_gt) for x in abs_]
            for i in range(gt.shape[0]):
                rec = ([v for im in imgs for v in psnr_ssim(im[i], gt[i])] + [e[i] for e in errs]
                       + [L_rmse[i]])
                stats.setdefault(terrain[i], []).append(rec)
                allrecs.append(rec)

    res = summarize(allrecs)
    up = UPSTREAM[args.model]
    print("\n" + "=" * 72)
    print(f"Model {args.model}  -  {res['n']} images  -  {desc}")
    print("=" * 72)
    print(f"  {'':38s}{'PSNR dB':>10s}{'SSIM':>9s}{'ab RMSE':>10s}")
    for r in ROWS:
        label = LABELS[r] + (" [round trip]" if r == "oracle" and args.model == 1 else "")
        print(f"  {label:38s}{res[r]['psnr']:10.4f}{res[r]['ssim']:9.4f}{res[r]['ab_rmse']:10.2f}")
    print(f"  brightness error (L RMSE vs optical L, Lab units): {res['L_rmse']:.2f}")
    m = res["model"]
    for name, key in (("no color  ", "no_color"), ("mean color", "mean_color")):
        b = res[key]
        print(f"  colorizer vs {name}: {m['psnr'] - b['psnr']:+.2f} dB PSNR, "
              f"{m['ssim'] - b['ssim']:+.4f} SSIM, ab error {b['ab_rmse']:.2f} -> {m['ab_rmse']:.2f}")
    print(f"  upstream ColorSAR (different data/split, context only): PSNR {up[0]}  SSIM {up[1]}")

    per_terrain = {t: summarize(v) for t, v in sorted(stats.items())}
    if args.per_terrain:
        print(f"\n  {'terrain':<12}{'n':>6}{'PSNR':>9}{'no-color':>10}{'mean-col':>10}"
              f"{'SSIM':>8}{'ab RMSE':>9}{'no-color':>10}{'mean-col':>10}")
        for t, s in per_terrain.items():
            print(f"  {t:<12}{s['n']:>6}{s['model']['psnr']:>9.3f}{s['no_color']['psnr']:>10.3f}"
                  f"{s['mean_color']['psnr']:>10.3f}{s['model']['ssim']:>8.4f}"
                  f"{s['model']['ab_rmse']:>9.2f}{s['no_color']['ab_rmse']:>10.2f}"
                  f"{s['mean_color']['ab_rmse']:>10.2f}")

    if args.out:
        result = {"model": args.model, "checkpoint": args.checkpoint, "eval_set": desc,
                  "split": args.split, "fingerprint": meta["fingerprint"],
                  "denoised": dncnn is not None, **res, "per_terrain": per_terrain,
                  "upstream_reference": {"psnr": up[0], "ssim": up[1],
                                         "note": "different data selection/split"}}
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved metrics to {args.out}")


if __name__ == "__main__":
    main()
