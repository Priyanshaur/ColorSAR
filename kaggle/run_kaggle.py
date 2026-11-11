"""
ColorSAR - Kaggle GPU job: train Model 3 + Model 1b, evaluate everything on held-out pairs.

Runs as a Kaggle "script" kernel (see kernel-metadata.json) with two inputs attached:
    requiemonk/sentinel12-image-pairs-segregated-by-terrain   the dataset (v_2)
    <user>/colorsar-code                                      this repo, flattened by
                                                              kaggle/prepare_code.py
Everything it produces lands in /kaggle/working/results (download with
`kaggle kernels output <user>/colorsar-train -p kaggle_out`).
"""
import glob
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working/ColorSAR"
RESULTS = "/kaggle/working/results"


def find_dir(predicate, root="/kaggle/input"):
    for d, dirs, files in os.walk(root):
        if predicate(d, dirs, files):
            return d
    return None


def run(name, args, log):
    t0 = time.time()
    print(f"\n===== {name}: {' '.join(args)}", flush=True)
    with open(log, "w") as f:
        p = subprocess.Popen([sys.executable, "-u"] + args, cwd=WORK, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if "Warning" not in line:
                print(line, end="", flush=True)
            f.write(line)
        p.wait()
    print(f"===== {name}: exit {p.returncode} after {(time.time()-t0)/60:.1f} min", flush=True)
    return p.returncode == 0


if __name__ == "__main__":
    import torch
    print("torch", torch.__version__, "| GPUs:",
          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    data_root = find_dir(lambda d, dirs, f: "s1" in dirs and "s2" in dirs)
    data_root = os.path.dirname(data_root)
    code = find_dir(lambda d, dirs, files: "train_colorizer.py" in files)
    assert data_root and code, (data_root, code)
    print("data:", data_root, "| code:", code)

    # rebuild the repo layout from the flat code dataset ("model2_run__dncnn.pth" -> model2_run/)
    os.makedirs(WORK, exist_ok=True)
    for src in glob.glob(os.path.join(code, "*")):
        name = os.path.basename(src)
        dst = os.path.join(WORK, *name.split("__"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    os.makedirs(RESULTS, exist_ok=True)
    R = ["--root", data_root, "--num_workers", "4"]
    L = lambda n: os.path.join(RESULTS, n + ".log")  # noqa: E731

    ok3 = run("train model 3", ["Train_model3.py", "--out", "model3_run"] + R, L("train_model3"))
    ok1b = run("train model 1b", ["Train_model1.py", "--out", "model1b_run", "--loss", "bins",
                                  "--train_adapters", "--imagenet_norm", "--augment"] + R,
               L("train_model1b"))

    E = ["evaluate.py", "--per_terrain", "--batch_size", "32"] + R
    if ok3:
        run("eval model 3 (canonical test)",
            E + ["--model", "3", "--checkpoint", "model3_run/model_3.pth",
                 "--out", "model3_run/metrics.json"], L("eval_model3"))
        run("eval model 3 (pairs unseen by Model 2 too)",
            E + ["--model", "3", "--checkpoint", "model3_run/model_3.pth",
                 "--intersect", "model2_run/split.json",
                 "--out", "model3_run/metrics_vs_model2.json"], L("eval_model3_vs2"))
        run("eval model 2 (same pairs)",
            E + ["--model", "2", "--checkpoint", "model2_run/model_2.pth",
                 "--intersect", "model3_run/split.json",
                 "--out", "model3_run/metrics_model2_same_pairs.json"], L("eval_model2_vs3"))
    if ok1b:
        run("eval model 1b (canonical test)",
            E + ["--model", "1", "--checkpoint", "model1b_run/model_1.pth",
                 "--out", "model1b_run/metrics.json"], L("eval_model1b"))
        run("eval model 1b (pairs Model 1 did not train on)",
            E + ["--model", "1", "--checkpoint", "model1b_run/model_1.pth",
                 "--intersect", "model1_run/split.json",
                 "--out", "model1b_run/metrics_vs_model1.json"], L("eval_model1b_vs1"))
        run("eval model 1 (same pairs)",
            E + ["--model", "1", "--checkpoint", "model1_run/model_1.pth",
                 "--intersect", "model1b_run/split.json",
                 "--out", "model1b_run/metrics_model1_same_pairs.json"], L("eval_model1_vs1b"))

    for run_dir in ("model3_run", "model1b_run"):
        src = os.path.join(WORK, run_dir)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(RESULTS, run_dir), dirs_exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)   # keep the kernel output small
    print("\nresults:", sorted(os.listdir(RESULTS)))
