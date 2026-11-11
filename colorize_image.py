"""
ColorSAR - Model 1 colorization of a single image (kept for backward compatibility).

This used to be a separate implementation with its own preprocessing; it now simply
runs colorize.py in "gray" (Model 1) mode so there is exactly one inference path.

Usage:
    python colorize_image.py --image sample_input.jpeg --checkpoint model1_run/model_1.pth
"""
import argparse
import sys

import colorize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="sample_input.jpeg")
    ap.add_argument("--checkpoint", default="model1_run/model_1.pth")
    ap.add_argument("--out", default="colorized_result.png")
    args = ap.parse_args()
    sys.argv = [sys.argv[0], "--image", args.image, "--checkpoint", args.checkpoint,
                "--out", args.out, "--mode", "gray"]
    colorize.main()
