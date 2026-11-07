"""
ColorSAR - Visual Sanity Check
Runs a REAL image from the dataset through the full (untrained) encoder-decoder
architecture and saves the predicted colorization as a PNG.

Since the model is randomly initialized (no training yet), the output will look
like colored noise/static -- that's expected. This confirms the full pipeline
(load -> Lab -> patch -> encoder -> decoder -> reconstruct -> save) works
end-to-end on real data, which is what matters for tonight.

Usage:
    python visual_test.py "v_2/agri/s2/some_image.png"
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights
from PIL import Image
import cv2
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Data pipeline (same as data_pipeline.py) ----------
def rgb_to_lab(img):
    img_lab = rgb2lab(img).astype("float32")
    img_lab = np.transpose(img_lab, (2, 0, 1))
    L_channel = img_lab[[0], ...]
    ab_channels = img_lab[[1, 2], ...]
    L_channel = 2 * (L_channel - 0) / (100 - 0) - 1
    ab_channels = 2 * (ab_channels - (-128)) / (127 - (-128)) - 1
    return L_channel, ab_channels


def load_and_preprocess(image_path, size=224):
    color_img = np.array(Image.open(image_path).convert("RGB"))
    color_img = cv2.resize(color_img, (size, size))
    L_channel, ab_channels = rgb_to_lab(color_img)
    return color_img, L_channel, ab_channels


# ---------- Model (same as model_architecture.py) ----------
class EnsembleEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # Pinned (not DEFAULT) + seeded frozen layers: see model_architecture.py
        weights_r = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        weights_d = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

        self.resnet50 = models.resnet50(weights=weights_r)
        self.densenet121 = models.densenet121(weights=weights_d)

        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.densenet121.classifier = nn.Identity()

        # Frozen-but-random layers must be identical everywhere, or decoder-only
        # checkpoints cannot be loaded on another machine. See model_architecture.
        _rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.conv1x1_resnet50 = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=1), nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(1024, 512, kernel_size=1), nn.Conv2d(2048, 1024, kernel_size=1)
        ])
        self.conv1x1_densenet121 = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=1), nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(1024, 512, kernel_size=1), nn.Conv2d(1024, 1024, kernel_size=1)
        ])
        self.fusion_blocks = nn.ModuleList([
            self.fusion_block(128, 128), self.fusion_block(256, 256),
            self.fusion_block(512, 512), self.fusion_block(1024, 1024)
        ])
        torch.set_rng_state(_rng_state)

    def fusion_block(self, r, d):
        return nn.Sequential(nn.Conv2d(r + d, r, kernel_size=1), nn.BatchNorm2d(r), nn.ReLU(inplace=True))

    def forward(self, x):
        resnet_features = []
        ri = x
        for i, layer in enumerate(self.resnet50.children()):
            ri = layer(ri)
            if i in [4, 5, 6, 7]:
                resnet_features.append(self.conv1x1_resnet50[i - 4](ri))

        densenet_features = []
        idx = 0
        di = x
        for i, layer in enumerate(self.densenet121.features):
            di = layer(di)
            if i in [4, 6, 8, 11]:
                densenet_features.append(self.conv1x1_densenet121[idx](di))
                idx += 1

        fused = []
        for i in range(4):
            f = torch.cat((resnet_features[i], densenet_features[i]), dim=1)
            fused.append(self.fusion_blocks[i](f))
        return fused


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decode1 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.decode2 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.decode3 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.decode4 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.decode5 = nn.Sequential(nn.Conv2d(64, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.Tanh(),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    def forward(self, f7, f14, f28, f56):
        x = self.decode1(f7)
        x = torch.cat([x, f14], dim=1)
        x = self.decode2(x)
        x = torch.cat([x, f28], dim=1)
        x = self.decode3(x)
        x = torch.cat([x, f56], dim=1)
        x = self.decode4(x)
        return self.decode5(x)


class ColorizationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        f56, f28, f14, f7 = self.encoder(x)
        return self.decoder(f7, f14, f28, f56)


if __name__ == "__main__":
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    assert test_image, "Pass an image path, e.g. python visual_test.py v_2/agri/s2/xxx.png"

    print(f"Loading real image: {test_image}")
    real_rgb, L_channel, ab_channels_true = load_and_preprocess(test_image)

    print("Building model (untrained / random weights)...")
    encoder = EnsembleEncoder(pretrained=False).to(device)
    decoder = Decoder().to(device)
    model = ColorizationModel(encoder, decoder)
    model.eval()

    # model expects 3-channel input; repeat L across 3 channels (as grayscale->3ch)
    L_input = torch.from_numpy(L_channel).unsqueeze(0)          # (1,1,224,224)
    L_input_3ch = L_input.repeat(1, 3, 1, 1).to(device)          # (1,3,224,224)

    print("Running forward pass...")
    with torch.no_grad():
        pred_ab = model(L_input_3ch).cpu().numpy()[0]            # (2,224,224), in [-1,1]

    # unscale predicted ab and true L back to Lab ranges
    L_unscaled = (L_channel + 1) * 0.5 * 100                     # (1,224,224)
    ab_pred_unscaled = (pred_ab + 1) * 0.5 * 255 - 128            # (2,224,224)

    lab_pred = np.concatenate([L_unscaled, ab_pred_unscaled], axis=0)
    lab_pred = np.transpose(lab_pred, (1, 2, 0))
    rgb_pred = lab2rgb(lab_pred)

    # side-by-side figure: real optical image vs untrained-model "colorized" output
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(real_rgb)
    axes[0].set_title("Real Optical Image (ground truth)")
    axes[0].axis("off")
    axes[1].imshow(rgb_pred)
    axes[1].set_title("Model Output (UNTRAINED)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("visual_test_output.png", dpi=150)
    print("\nSaved: visual_test_output.png")
    print("This is expected to look like noise -- the model has random weights.")
    print("This confirms the FULL pipeline (data -> encoder -> decoder -> image) runs end-to-end on real data.")