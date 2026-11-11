"""
ColorSAR - regenerate a run's figures from its saved history and checkpoint.

    python make_figures.py --model 1            # model1_run/loss_curve.png + sample_predictions.png
    python make_figures.py --model 2

Use this whenever a checkpoint is replaced (e.g. one trained on Kaggle) so the figures
in the run folder always show the checkpoint that is actually there. Sample images are
drawn from the checkpoint's held-out pairs (splits.heldout).
"""
import argparse
import json
import os

import matplotlib
import torch
from torch.utils.data import DataLoader

from color_utils import PairDataset, find_pairs
from dncnn import load_dncnn
from model_architecture import load_colorizer
from splits import heldout, subsample
from train_colorizer import Colorizer, save_samples

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def loss_curve(history_path, out_path, title):
    with open(history_path) as f:
        h = json.load(f)
    ep = range(1, len(h["val_loss"]) + 1)
    best = min(range(len(h["val_loss"])), key=h["val_loss"].__getitem__) + 1
    plt.figure(figsize=(7, 4))
    plt.plot(ep, h["train_loss"], marker="o", ms=3, label="train")
    plt.plot(ep, h["val_loss"], marker="s", ms=3, label="val")
    plt.axvline(best, ls=":", c="gray", label=f"best val {min(h['val_loss']):.5f} (epoch {best})")
    plt.xlabel("epoch"); plt.ylabel("MSE (normalized a/b)"); plt.title(title)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=130); plt.close()
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int, choices=[1, 2], required=True)
    ap.add_argument("--run", default=None, help="run folder (default model<k>_run)")
    ap.add_argument("--root", default="v_2")
    args = ap.parse_args()
    run = args.run or f"model{args.model}_run"

    loss_curve(os.path.join(run, "history.json"), os.path.join(run, "loss_curve.png"),
               f"Model {args.model} training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, dec, meta = load_colorizer(os.path.join(run, f"model_{args.model}.pth"), device)
    dncnn = load_dncnn(os.path.join(run, "dncnn.pth"), device) if args.model == 2 else None
    if dncnn is not None and meta["dncnn_res"] != "native":
        # save_samples applies the DnCNN at native resolution; legacy checkpoints expect
        # resize-then-DnCNN, so wrap it to reproduce exactly that input.
        from color_utils import resize_t

        class _Legacy(torch.nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, x):
                return self.net(resize_t(x))
        dncnn = _Legacy(dncnn)
    pairs, desc = heldout(find_pairs(args.root), os.path.join(run, "split.json"))
    print(f"samples from: {desc}")
    loader = DataLoader(PairDataset(subsample(pairs, 4, seed=7), load_sar=args.model == 2),
                        batch_size=4)
    save_samples(loader, Colorizer(enc, dec, meta["input_norm"]), args.model, dncnn, device,
                 os.path.join(run, "sample_predictions.png"))
    print(f"[OK] {os.path.join(run, 'sample_predictions.png')}")
