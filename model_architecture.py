"""
ColorSAR - Model Architecture (Ensemble Encoder + Decoder)
Extracted and verified from original_model.ipynb (cells 15, 17, 19)
Source: https://github.com/nishant9083/ColorSAR
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for the encoder's randomly-initialised-but-frozen layers. See the comment
# inside EnsembleEncoder.__init__ -- this constant is what makes decoder-only
# checkpoints portable between machines.
ENCODER_FROZEN_SEED = 42


def encoder_fingerprint(encoder):
    """Short hash of the encoder's frozen random layers.

    Print this while training and again while evaluating: identical values mean a
    decoder-only checkpoint is being loaded against the very encoder it was
    trained with. Different values = the metrics you are about to compute are
    meaningless (the decoder sees features it has never seen before).
    """
    import hashlib
    h = hashlib.sha256()
    for part in (encoder.conv1x1_resnet50, encoder.conv1x1_densenet121,
                 encoder.fusion_blocks):
        for m in part:
            for v in m.state_dict().values():
                h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


class EnsembleEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(EnsembleEncoder, self).__init__()

        # Pinned explicitly instead of *_Weights.DEFAULT: DEFAULT silently changed
        # from IMAGENET1K_V1 to IMAGENET1K_V2 across torchvision releases, and the
        # two backbones produce features ~3.5x apart -- a decoder trained against
        # one is unusable with the other.
        weights_r = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        weights_d = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

        self.resnet50 = models.resnet50(weights=weights_r)
        self.densenet121 = models.densenet121(weights=weights_d)

        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.densenet121.classifier = nn.Identity()

        # The 1x1 projections and fusion blocks below have no pretrained weights:
        # they start random and stay frozen for the whole training run. Checkpoints
        # store only the decoder, so these layers MUST come out bit-identical on
        # every machine -- otherwise a shipped checkpoint scores far below the val
        # loss it was trained to (measured: 0.045 instead of 0.0043, because the
        # features differ by ~116% between two random draws).
        # Re-seed right before drawing them so the result depends only on
        # ENCODER_FROZEN_SEED and not on how much RNG torchvision consumed above;
        # the surrounding RNG state is restored so training stays unaffected.
        _rng_state = torch.get_rng_state()
        torch.manual_seed(ENCODER_FROZEN_SEED)

        self.conv1x1_resnet50 = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(2048, 1024, kernel_size=1)
        ])

        self.conv1x1_densenet121 = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.Conv2d(1024, 1024, kernel_size=1)
        ])

        self.fusion_blocks = nn.ModuleList([
            self.fusion_block(128, 128),
            self.fusion_block(256, 256),
            self.fusion_block(512, 512),
            self.fusion_block(1024, 1024)
        ])

        torch.set_rng_state(_rng_state)

    def fusion_block(self, in_channels_resnet, in_channels_densenet):
        return nn.Sequential(
            nn.Conv2d(in_channels_resnet + in_channels_densenet, in_channels_resnet, kernel_size=1),
            nn.BatchNorm2d(in_channels_resnet),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        resnet_features = []
        resnet_input = x
        for i, layer in enumerate(self.resnet50.children()):
            resnet_input = layer(resnet_input)
            if i in [4, 5, 6, 7]:
                resnet_features.append(self.conv1x1_resnet50[i-4](resnet_input))

        densenet_features = []
        idx = 0
        densenet_input = x
        for i, layer in enumerate(self.densenet121.features):
            densenet_input = layer(densenet_input)
            if i in [4, 6, 8, 11]:
                densenet_features.append(self.conv1x1_densenet121[idx](densenet_input))
                idx += 1

        fused_features = []
        for i in range(4):
            fused = torch.cat((resnet_features[i], densenet_features[i]), dim=1)
            fused = self.fusion_blocks[i](fused)
            fused_features.append(fused)

        return fused_features


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decode1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decode3 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decode4 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decode5 = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2), nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, features_7x7, features_14x14, features_28x28, features_56x56):
        x = self.decode1(features_7x7)
        x = torch.cat([x, features_14x14], dim=1)
        x = self.decode2(x)
        x = torch.cat([x, features_28x28], dim=1)
        x = self.decode3(x)
        x = torch.cat([x, features_56x56], dim=1)
        x = self.decode4(x)
        output = self.decode5(x)
        return output


class ColorizationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ColorizationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features_56x56, features_28x28, features_14x14, features_7x7 = self.encoder(x)
        output = self.decoder(features_7x7, features_14x14, features_28x28, features_56x56)
        return output


if __name__ == "__main__":
    print(f"Device: {device}")
    print("Building EnsembleEncoder (random init, pretrained=False for a fast structural test)...")
    encoder = EnsembleEncoder(pretrained=False).to(device)
    decoder = Decoder().to(device)
    model = ColorizationModel(encoder, decoder)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}  (expected [1, 2, 224, 224] -> predicted a,b channels)")

    assert output.shape == (1, 2, 224, 224), "Output shape mismatch!"
    print("\nMODEL ARCHITECTURE VERIFIED: forward pass runs correctly, output shape matches spec.")
