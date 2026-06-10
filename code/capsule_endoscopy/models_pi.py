"""Physics-informed model variants (5-channel input).

The first conv of each ImageNet-pretrained backbone is replaced with a
Conv2d of in_channels = 3 + extra_channels. Pretrained weights are copied
into the first 3 input positions; the remaining positions are zero-initialized
so the model starts behaviorally identical to the 3-channel RGB baseline and
only diverges if the physics channels carry useful signal.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = {
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
}


def _expand_first_conv(conv: nn.Conv2d, extra_channels: int) -> nn.Conv2d:
    new = nn.Conv2d(
        in_channels=conv.in_channels + extra_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : conv.in_channels] = conv.weight
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_backbone_pi(model_name: str, pretrained: bool = True,
                      extra_channels: int = 2) -> Tuple[nn.Module, int]:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {sorted(SUPPORTED_MODELS)}")

    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.conv1 = _expand_first_conv(m.conv1, extra_channels)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        return m, in_features

    if model_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.conv1 = _expand_first_conv(m.conv1, extra_channels)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
        return m, in_features

    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.features[0][0] = _expand_first_conv(m.features[0][0], extra_channels)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, in_features

    if model_name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.features[0][0] = _expand_first_conv(m.features[0][0], extra_channels)
        in_features = m.classifier[2].in_features
        # torchvision's ConvNeXt puts the Flatten inside .classifier; if we
        # replace the whole classifier with Identity the output stays
        # (B, C, 1, 1) and downstream Linear heads crash with
        # "mat1 and mat2 shapes cannot be multiplied". Keep LayerNorm2d
        # (classifier[0]) and Flatten (classifier[1]); drop the final Linear.
        m.classifier = nn.Sequential(m.classifier[0], m.classifier[1])
        return m, in_features

    raise RuntimeError("Unexpected model name")


class ImageClassifierPI(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True,
                 extra_channels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.backbone, feat_dim = build_backbone_pi(
            model_name, pretrained=pretrained, extra_channels=extra_channels,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class MultiTaskClassifierPI(nn.Module):
    def __init__(self, model_name: str, num_triage_classes: int, num_subtype_classes: int,
                 pretrained: bool = True, extra_channels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.backbone, feat_dim = build_backbone_pi(
            model_name, pretrained=pretrained, extra_channels=extra_channels,
        )
        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.triage_head = nn.Linear(feat_dim, num_triage_classes)
        self.subtype_head = nn.Linear(feat_dim, num_subtype_classes)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.shared(feat)
        return self.triage_head(feat), self.subtype_head(feat)


class _BackboneWithFeatureMap(nn.Module):
    """Wraps a torchvision backbone to expose both the pooled vector
    [B, feat_dim] and the pre-pool spatial feature map [B, feat_dim, h, w].

    Used by ImageClassifierPIDistill so the auxiliary segmentation head can
    predict a P_blood map from the spatial features. 3-channel RGB input only
    — this is the deployment-friendly variant.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {sorted(SUPPORTED_MODELS)}")

        if model_name == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.body = nn.Sequential(m.layer1, m.layer2, m.layer3, m.layer4)
            self.pool = m.avgpool
            self.feat_dim = m.fc.in_features
        elif model_name == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
            self.body = nn.Sequential(m.layer1, m.layer2, m.layer3, m.layer4)
            self.pool = m.avgpool
            self.feat_dim = m.fc.in_features
        elif model_name == "efficientnet_b0":
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Identity()
            self.body = m.features
            self.pool = m.avgpool
            self.feat_dim = m.classifier[1].in_features
        elif model_name == "convnext_tiny":
            m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Identity()
            self.body = m.features
            self.pool = m.avgpool
            self.feat_dim = m.classifier[2].in_features

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        feat_map = self.body(x)             # [B, C, h, w]  (h, w = input/32)
        pooled = self.pool(feat_map).flatten(1)  # [B, C]
        return pooled, feat_map


class ImageClassifierPIDistill(nn.Module):
    """3-channel RGB classifier with an auxiliary P_blood-prediction head.

    Trained with total_loss = CE(class_logits, label) + lambda * MSE(predicted_pblood, P_blood_target),
    where P_blood_target is computed analytically from the input RGB by
    physics_prior.blood_probability (treated as a teacher signal).

    Why this exists: the deployed model uses standard 3-channel RGB input
    and does not need the physics channels at inference time. The backbone
    has internalized hemoglobin-aware representations during training. The
    decoder also produces a free interpretability heatmap.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.2, decoder_hidden: int = 64):
        super().__init__()
        self.backbone = _BackboneWithFeatureMap(model_name, pretrained=pretrained)
        feat_dim = self.backbone.feat_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.distill_decoder = nn.Sequential(
            nn.Conv2d(feat_dim, decoder_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_hidden, decoder_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_hidden, 1, kernel_size=1),
        )
        # Inference-time toggle: when True, forward() skips the decoder so the
        # deployed model takes plain 3-channel RGB and pays no decoder cost.
        # Set via `model.deploy_mode = True` after loading the checkpoint.
        self.deploy_mode = False

    def forward(self, x: torch.Tensor):
        """In training: returns (class_logits, pblood_logits).
        In deploy mode (`self.deploy_mode = True`): returns class_logits only.

        pblood_logits are pre-sigmoid; the training script applies
        BCEWithLogitsLoss against the sigmoid-form target for numerical
        stability. h,w = input_size / 32.
        """
        pooled, feat_map = self.backbone(x)
        logits = self.head(pooled)
        if self.deploy_mode:
            return logits
        pblood_logits = self.distill_decoder(feat_map)
        return logits, pblood_logits
