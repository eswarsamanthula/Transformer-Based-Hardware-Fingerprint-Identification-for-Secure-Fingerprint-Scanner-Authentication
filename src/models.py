"""
models.py — builds any model named in config.MODEL_REGISTRY.

This is the ONLY file where this project's work differs from the base
paper: it adds 8 transformer/hybrid-CNN backbones — ViT-B/16, Swin-Tiny,
DeiT-Small, ConvNeXt-Tiny, TinyViT-5M, MobileViT-XS, EfficientFormer-L1,
and PiT-Ti — alongside the paper's original 6 CNNs (EfficientNetB3,
MobileNet, ResNet50, ResNet152, VGG16, LeNet5), loaded through the same
`timm` interface so training/eval code treats every architecture
identically.

Every model is built with pretrained=False (see config.py) to match the
paper's "all models were trained from scratch." Be aware of the practical
consequence: a ViT-B/16 (86M params) or ResNet152 trained from random
weights on ~1,400 images may fail to converge, collapse to predicting one
class, or badly overfit — that's an expected risk of this setup, not a
bug in this code. Watch train.py's per-epoch val_acc; if a model is stuck
at 1/NUM_SCANNERS accuracy after ~15-20 epochs, it isn't learning and no
amount of extra epochs will fix it. If that happens, the honest fallback
is pretrained=True for that specific model, reported as a deviation.
This project has since applied that fallback for real: the official
EfficientFormer-L1 (RGB) run trains from ImageNet-pretrained weights
(see config.py) — the one deliberate deviation from the paper protocol.
"""

import torch
import torch.nn as nn
import timm

from config import MODEL_REGISTRY


class LeNet5(nn.Module):
    """
    The paper includes LeNet5 as a lightweight baseline. It predates timm
    and isn't in the library, so it's implemented directly here, sized for
    grayscale or RGB input.

    Per the paper: the fully connected section was modified to add a hidden
    dense layer of 1024 neurons, another hidden dense layer of 512 neurons,
    and an output layer of 10 neurons with softmax. All hidden layers use
    ReLU; a dropout layer of factor 0.5 sits between each FC layer.
    """

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class TimmFeatureExtractor(nn.Module):
    """
    Wraps a timm backbone built with num_classes=0 (feature extractor only —
    timm keeps its global pooling) plus the paper's custom classification
    head. Used for models where the paper explicitly overrides the head:
    EfficientNetB3 and VGG16, both of which get
        Linear(*, 512) -> ReLU -> Dropout(0.2) -> Linear(512, num_classes)
    per Section VI (GAP -> 512-neuron hidden dense -> dropout 0.2 -> 10).
    """

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LazyLinear(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(model_name: str, num_classes: int, in_channels: int) -> nn.Module:
    """
    num_classes: 10 for multiclass scanner identification, 1 for binary
                 (one-vs-all) scanner verification — caller decides which.
    in_channels: 1 (grayscale) or 3 (rgb) — set by the channel mode being run.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY)}")

    cfg = MODEL_REGISTRY[model_name]

    if model_name == "lenet5":
        return LeNet5(num_classes=num_classes, in_channels=in_channels)

    if model_name in ("efficientnet_b3", "vgg16"):
        # Paper's custom head, not timm's default — see TimmFeatureExtractor.
        backbone = timm.create_model(
            cfg["timm_name"],
            pretrained=cfg["pretrained"],
            num_classes=0,  # feature extractor only; timm retains its global pooling
            in_chans=in_channels,
        )
        if model_name == "vgg16":
            # Paper: "the top removed" — drop timm's pre_logits block (the
            # original 4096->4096 FC layers kept as conv). Without this the
            # head would consume 4096 features (~136M params) instead of the
            # paper's pool-then-512 design (~15M params).
            backbone.pre_logits = nn.Identity()
        return TimmFeatureExtractor(backbone, num_classes=num_classes)

    # All other models: timm's default num_classes=N classification head.
    model = timm.create_model(
        cfg["timm_name"],
        pretrained=cfg["pretrained"],
        num_classes=num_classes,
        in_chans=in_channels,
    )
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # sanity check: build every registered model, both channel modes
    for name in MODEL_REGISTRY:
        for ch in (1, 3):
            m = build_model(name, num_classes=10, in_channels=ch)
            x = torch.randn(1, ch, 224, 224) if name != "lenet5" else torch.randn(1, ch, 32, 32)
            try:
                out = m(x)
                print(f"{name:16s} ch={ch}  out={tuple(out.shape)}  params={count_params(m):,}")
            except Exception as e:
                print(f"{name:16s} ch={ch}  FAILED: {e}")
