"""
ColorSAR - DnCNN denoiser (Model 2 front-end)

Extracted from the original gray_model.ipynb:
    depth=17, n_channels=64, residual learning  ->  output = x - net(x)

In the original project DnCNN is trained to map the SAR (s1) grayscale image
onto the matching optical (s2) grayscale image, i.e. it strips radar speckle
and produces the luminance that the colorizer then consumes.

Usage:
    from dncnn import DnCNN, load_dncnn
"""
import os

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(1, n_channels, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(n_channels, 1, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)


def load_dncnn(path, device):
    """Load a DnCNN checkpoint (state_dict or whole module) and put it in eval mode."""
    assert os.path.exists(path), (
        f"DnCNN checkpoint not found at: {path}\n"
        "Train it first:  python train_dncnn.py   (writes model2_run/dncnn.pth)"
    )
    model = DnCNN()
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        model.load_state_dict(obj)
    else:
        model = obj
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


if __name__ == "__main__":
    net = DnCNN()
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.rand(2, 1, 256, 256)
    y = net(x)
    print(f"DnCNN parameters: {n_params:,}")
    print(f"Input {tuple(x.shape)} -> output {tuple(y.shape)} (residual learning: x - net(x))")
    assert y.shape == x.shape
    print("\nDnCNN VERIFIED: forward pass runs, output shape matches input.")
