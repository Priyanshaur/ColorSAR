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

# Fingerprint of the seeded adapter layers that the shipped (format 1, decoder-only)
# checkpoints were trained against. Loading such a checkpoint against any other
# encoder is refused (see load_colorizer) instead of silently producing bad colors.
LEGACY_FINGERPRINT = "788e0728e71f"

# The encoder layers that have no pretrained weights ("adapters"). Format 2
# checkpoints store them next to the decoder, so they never depend on RNG state.
ADAPTER_NAMES = ("conv1x1_resnet50", "conv1x1_densenet121", "fusion_blocks")

# ImageNet statistics, used when a checkpoint was trained with input_norm="imagenet".
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
    """U-Net decoder.

    out_channels=None : the original head (2 channels, BatchNorm + tanh) -> normalized a/b.
    out_channels=k    : a raw k-channel head (no BN / activation), interpreted by
                        pipeline.decode_output according to the checkpoint's meta
                        (optional L channel + a/b regression or a/b bin logits).
    """

    def __init__(self, out_channels=None):
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
        if out_channels is None:
            self.decode5 = nn.Sequential(
                nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2), nn.Tanh(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            )
        else:
            self.decode5 = nn.Sequential(
                nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
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


# --------------------------------------------------------------------------- helpers
def encoder_input(L, input_norm="none"):
    """Normalized L (N,1,H,W) in [-1,1] -> 3-channel encoder input.

    "none"     : L repeated to 3 channels, as the original ColorSAR notebooks do
                 (and as every shipped checkpoint was trained).
    "imagenet" : L mapped to [0,1] gray and normalized with ImageNet mean/std, i.e.
                 the input statistics the frozen backbones were pretrained on.
    """
    x = L.repeat(1, 3, 1, 1)
    if input_norm == "none":
        return x
    if input_norm == "imagenet":
        x = (x + 1.0) * 0.5
        mean = torch.tensor(_IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std
    raise ValueError(f"unknown input_norm: {input_norm!r}")


def adapter_parameters(encoder):
    return [p for name in ADAPTER_NAMES for p in getattr(encoder, name).parameters()]


def freeze_encoder(encoder, train_adapters=False):
    """Freeze the pretrained backbones; optionally leave the adapters trainable."""
    for p in encoder.parameters():
        p.requires_grad = False
    for p in adapter_parameters(encoder) if train_adapters else []:
        p.requires_grad = True
    encoder_mode(encoder, training=False, train_adapters=train_adapters)


def encoder_mode(encoder, training, train_adapters):
    """Backbones always in eval mode (frozen BN stats); adapters follow `training`."""
    encoder.eval()
    if training and train_adapters:
        for name in ADAPTER_NAMES:
            getattr(encoder, name).train()


def save_checkpoint(path, encoder, decoder, meta):
    """Format 2 checkpoint: decoder + encoder adapters + metadata, all in one file.

    `encoder`/`decoder` may be wrapped in DataParallel; the bare modules are saved.
    """
    enc = getattr(encoder, "module", encoder)
    dec = getattr(decoder, "module", decoder)
    torch.save({
        "format": 2,
        "decoder": dec.state_dict(),
        "adapters": {name: getattr(enc, name).state_dict() for name in ADAPTER_NAMES},
        "meta": dict(meta, fingerprint=encoder_fingerprint(enc)),
    }, path)


def _strip_prefixes(state):
    out = {}
    for k, v in state.items():
        k = k[len("module."):] if k.startswith("module.") else k
        k = k[len("decoder."):] if k.startswith("decoder.") else k
        out[k] = v
    return out


def load_colorizer(path, device, pretrained=True, encoder=None):
    """Build (encoder, decoder, meta) from a checkpoint, ready for inference.

    Format 2 (written by train_colorizer.py): the adapters come from the file, so the
    result is identical on every machine regardless of RNG or torchvision defaults.

    Format 1 (shipped model_1.pth / model_2.pth, decoder-only state_dicts, optionally
    with `module.`/`decoder.` prefixes): the adapters are rebuilt from the seed and
    must reproduce LEGACY_FINGERPRINT, otherwise loading fails loudly.

    Pass `encoder` to reuse an already-built backbone (its adapters are overwritten
    for format 2 checkpoints, so do not share one encoder between such models).
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location=device, weights_only=False)

    if encoder is None:
        encoder = EnsembleEncoder(pretrained=pretrained)
    encoder = encoder.to(device)

    if isinstance(obj, dict) and obj.get("format") == 2:
        meta = dict(obj["meta"])
        for name in ADAPTER_NAMES:
            getattr(encoder, name).load_state_dict(obj["adapters"][name])
        dec_state = obj["decoder"]
    else:
        state = obj.state_dict() if isinstance(obj, nn.Module) else obj
        dec_state = _strip_prefixes(state)
        fp = encoder_fingerprint(encoder)
        if fp != LEGACY_FINGERPRINT:
            raise RuntimeError(
                f"{path} is a decoder-only (format 1) checkpoint trained against encoder "
                f"fingerprint {LEGACY_FINGERPRINT}, but this machine builds {fp}. Its colors "
                "would be meaningless. Use the same torch version / ENCODER_FROZEN_SEED, "
                "or retrain with train_colorizer.py (format 2 stores the adapters).")
        meta = {"format": 1, "input_norm": "none", "dncnn_res": "224",
                "fingerprint": fp}

    meta.setdefault("input_norm", "none")
    meta.setdefault("dncnn_res", "224")
    meta.setdefault("ab_head", "tanh")
    meta.setdefault("predict_L", False)
    decoder = Decoder(decoder_out_channels(meta))
    decoder.load_state_dict(dec_state)
    decoder = decoder.to(device).eval()
    freeze_encoder(encoder)
    return encoder, decoder, meta


def decoder_out_channels(meta):
    """Decoder head size for a checkpoint's meta (None = the original 2-channel tanh head).

    meta["ab_head"]  : "tanh" (original head), "reg" (raw a/b, tanh applied when decoding)
                       or "bins" (logits over meta["bins"] color bins)
    meta["predict_L"]: also predict the L channel (first output channel)
    """
    head = meta.get("ab_head", "tanh")
    L = bool(meta.get("predict_L", False))
    if head == "tanh" and not L:
        return None
    if head == "tanh":
        raise ValueError("predict_L needs ab_head 'reg' or 'bins'")
    n_ab = 2 if head == "reg" else len(meta["bins"])
    return n_ab + (1 if L else 0)


@torch.no_grad()
def predict_ab(encoder, decoder, L, input_norm="none"):
    f56, f28, f14, f7 = encoder(encoder_input(L, input_norm))
    return decoder(f7, f14, f28, f56)


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
