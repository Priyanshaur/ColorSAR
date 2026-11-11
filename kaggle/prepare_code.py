"""
Build the flat folder that is uploaded as the private Kaggle dataset `<user>/colorsar-code`.

Kaggle datasets are flat here: subfolders are encoded in the file name with "__"
(model2_run/dncnn.pth -> model2_run__dncnn.pth); run_kaggle.py rebuilds the layout.

    python kaggle/prepare_code.py --user <kaggle username> --out <folder>
    kaggle datasets create -p <folder>            # first time (private by default)
    kaggle datasets version -p <folder> -m "msg"  # updates
"""
import argparse
import glob
import json
import os
import shutil

FILES = sorted(set(glob.glob("*.py")) - {"app.py", "_sharpness.py"}) + [
    "model1_run/model_1.pth", "model1_run/split.json",
    "model2_run/model_2.pth", "model2_run/dncnn.pth", "model2_run/split.json",
    "model2_run/dncnn_split.json",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in FILES:
        shutil.copy(f, os.path.join(args.out, f.replace("/", "__")))
    with open(os.path.join(args.out, "dataset-metadata.json"), "w") as fh:
        json.dump({"title": "colorsar-code", "id": f"{args.user}/colorsar-code",
                   "licenses": [{"name": "CC0-1.0"}]}, fh, indent=2)
    print(f"{len(FILES)} files -> {args.out}")
