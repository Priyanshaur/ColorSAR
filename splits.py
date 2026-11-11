"""
ColorSAR - train / val / test splits

Canonical split (used by every training script from now on)
    Each pair is assigned by a stable SHA-1 hash of "<terrain>/<s1 filename>":
    10% test, 10% val, 80% train. The assignment does not depend on RNG state,
    the machine, the file order or --n_images, so the DnCNN, Model 1 and Model 2
    all hold out the *same* test pairs and evaluate.py can score on images that
    no stage of the pipeline ever trained on.

split.json (written next to every checkpoint)
    {"train": [keys...], "val": [keys...], "rule": "...", "upstream": "<other split.json>"}
    evaluate.py scores on every pair that is in neither train nor val of the
    checkpoint *and* of the upstream stage (Model 2's DnCNN).

The shipped checkpoints predate this module; `python splits.py --legacy` rebuilds
their split.json files from the exact recipe each one was trained with.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

TEST_FRAC = 0.10
VAL_FRAC = 0.10


def pair_key(pair):
    """(sar_path, optical_path, terrain) -> 'terrain/<s1 filename>'."""
    return f"{pair[2]}/{os.path.basename(pair[0])}"


def _bucket(key):
    """Stable value in [0, 1) derived from the key only."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def canonical_split(pairs):
    """-> {"train": [...], "val": [...], "test": [...]} lists of pairs."""
    out = {"train": [], "val": [], "test": []}
    for p in pairs:
        u = _bucket(pair_key(p))
        name = "test" if u < TEST_FRAC else "val" if u < TEST_FRAC + VAL_FRAC else "train"
        out[name].append(p)
    return out


def subsample(pairs, n, seed=42):
    """Seeded random subset of size n (all of them when n <= 0 or n >= len)."""
    if n <= 0 or n >= len(pairs):
        return list(pairs)
    idx = np.random.RandomState(seed).permutation(len(pairs))[:n]
    return [pairs[i] for i in sorted(idx)]


def write_split(path, train, val, rule, upstream=None):
    data = {"rule": rule, "train": sorted(pair_key(p) for p in train),
            "val": sorted(pair_key(p) for p in val)}
    if upstream:
        data["upstream"] = upstream
    with open(path, "w") as f:
        json.dump(data, f, indent=0)


def read_split(path):
    with open(path) as f:
        return json.load(f)


def heldout(pairs, split_path):
    """Pairs never trained on by the checkpoint whose split.json is `split_path`.

    Follows the "upstream" link (Model 2 -> its DnCNN) so a pair counts as held out
    only if no stage saw it in train or val. If that leaves nothing (the shipped
    Model 1 trained on 90% and validated on the other 10% of the data), falls back
    to the checkpoint's own val set and says so - val was used to pick the best
    epoch, so it is a slightly optimistic estimate.

    Returns (pairs, description).
    """
    seen, val_only, path, chain = set(), None, split_path, []
    while path:
        s = read_split(path)
        seen.update(s["train"]); seen.update(s["val"])
        if val_only is None:
            val_only = set(s["val"])
        chain.append(os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path))
        up = s.get("upstream")
        path = os.path.join(os.path.dirname(path), up) if up else None

    test = [p for p in pairs if pair_key(p) not in seen]
    if test:
        return test, f"held-out test set ({len(test)} pairs unseen by {' + '.join(chain)})"
    val = [p for p in pairs if pair_key(p) in val_only]
    return val, (f"VALIDATION set ({len(val)} pairs) - no unseen pairs exist for this "
                 "checkpoint; val picked the best epoch, so this is slightly optimistic")


# --------------------------------------------------------------------------- legacy
def legacy_splits(pairs):
    """Rebuild the exact splits the shipped checkpoints were trained with.

    Model 1 (ColorSAR_Kaggle_Training.ipynb, original version): all 16,000 sorted s2
        paths, random_split 90/10 with torch seed 42.
    DnCNN (train_dncnn.py, original version): permutation(seed 42)[:10000], 80/20.
    Model 2 (Train_model2.py, original version): permutation(seed 42)[:8000], 80/20.
    """
    from torch.utils.data import random_split

    def rsplit(items, n_val):
        tr, va = random_split(range(len(items)), [len(items) - n_val, n_val],
                              generator=torch.Generator().manual_seed(42))
        return [items[i] for i in tr.indices], [items[i] for i in va.indices]

    by_s2 = sorted(pairs, key=lambda p: p[1])
    m1 = rsplit(by_s2, max(int(0.10 * len(by_s2)), 64))

    perm = np.random.RandomState(42).permutation(len(pairs))
    dn_sub = [pairs[i] for i in perm[:10000]]
    dn = rsplit(dn_sub, max(int(0.2 * len(dn_sub)), 1))
    m2_sub = [pairs[i] for i in perm[:8000]]
    m2 = rsplit(m2_sub, max(int(0.2 * len(m2_sub)), 1))
    return {"model1": m1, "dncnn": dn, "model2": m2}


if __name__ == "__main__":
    from color_utils import find_pairs

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="v_2")
    ap.add_argument("--legacy", action="store_true",
                    help="write split.json files for the shipped model1_run/model2_run checkpoints")
    args = ap.parse_args()

    pairs = find_pairs(args.root)
    assert pairs, f"No s1/s2 pairs found under {args.root}"
    c = canonical_split(pairs)
    print({k: len(v) for k, v in c.items()})

    if args.legacy:
        s = legacy_splits(pairs)
        rule = "legacy recipe reconstructed by splits.legacy_splits ({})"
        write_split("model1_run/split.json", *s["model1"], rule.format("Kaggle notebook 90/10"))
        write_split("model2_run/dncnn_split.json", *s["dncnn"], rule.format("train_dncnn 10000, 80/20"))
        write_split("model2_run/split.json", *s["model2"], rule.format("Train_model2 8000, 80/20"),
                    upstream="dncnn_split.json")
        for f in ("model1_run/split.json", "model2_run/split.json"):
            print(f, "->", heldout(pairs, f)[1])
