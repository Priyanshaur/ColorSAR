"""
ColorSAR - Kaggle GPU job: train + score the conditional pixel diffusion model
(train_diffusion.py) on the held-out scenes.

Inputs (kernel-metadata.json): the dataset (v_2) and <user>/colorsar-code
(kaggle/prepare_code.py). Output: /kaggle/working/results/diffusion_run/ and
train_diffusion.log. Push with  kaggle kernels push -p kaggle/diffusion --accelerator NvidiaTeslaT4

    MODE = "smoke"  the real model and batch sizes on a few hundred tiles for ~10 min
                    (catches crashes and out-of-memory before a long run)
    MODE = "full"   7 h of training on every training pair, then the full test scoring
    MODE = "small"  the same at a quarter of the width (base 32 channels), 2.5 h of training
                    (pushed as the separate kernel <user>/colorsar-diffusion-small)
"""
import glob
import os
import shutil
import subprocess
import sys
import time

MODE = "full"
WORK = "/kaggle/working/ColorSAR"
RESULTS = "/kaggle/working/results"
T0 = time.time()

ARGS = {
    "smoke": ["--hours", "0.15", "--full_frac", "0.5", "--n_train", "512", "--n_test", "96",
              "--val_every_min", "4", "--ddim_steps", "20", "--n_samples", "2",
              "--session_hours", "1.0"],
    "full": ["--hours", "7.0", "--session_hours", "11.3"],
    # quarter-width model (base 32 channels), shorter run: a fast second result
    "small": ["--hours", "2.5", "--session_hours", "4.5", "--ch", "32", "--lr", "2e-4",
              "--bs", "64", "--bs_full", "24"],
}
OUT_NAME = {"smoke": "diffusion_run", "full": "diffusion_run", "small": "diffusion_small_run"}


def find_dir(predicate, root="/kaggle/input"):
    for d, dirs, files in os.walk(root):
        if predicate(d, dirs, files):
            return d
    return None


if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips"], check=False)
    import torch
    print("mode", MODE, "| torch", torch.__version__, "| GPUs:",
          [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)

    data_root = os.path.dirname(find_dir(lambda d, dirs, f: "s1" in dirs and "s2" in dirs))
    code = find_dir(lambda d, dirs, files: "train_diffusion.py" in files)
    assert data_root and code, (data_root, code)
    print("data:", data_root, "| code:", code, flush=True)
    os.makedirs(WORK, exist_ok=True)
    for src in glob.glob(os.path.join(code, "*")):
        dst = os.path.join(WORK, *os.path.basename(src).split("__"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    out = os.path.join(RESULTS, OUT_NAME[MODE])
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "train_diffusion.py", "--root", data_root, "--out", out,
           "--workers", "4"] + ARGS[MODE]
    print("run:", " ".join(cmd), flush=True)
    with open(os.path.join(RESULTS, "train_diffusion.log"), "w") as log:
        p = subprocess.Popen(cmd, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if "Warning" not in line:
                print(line, end="", flush=True)
            log.write(line)
            log.flush()
        p.wait()
    print(f"exit {p.returncode} after {(time.time() - T0) / 3600:.2f} h", flush=True)
    shutil.rmtree(WORK, ignore_errors=True)
    print("results:", sorted(os.listdir(out)))
