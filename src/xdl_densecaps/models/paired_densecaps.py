"""Second-stage paired whole/detail DenseCaps classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from xdl_densecaps.models.backbone import DenseNet121FeatureExtractor
from xdl_densecaps.models.densenet_capsnet import RoutingCapsuleLayer


class DenseCapsEncoderBranch(nn.Module):
    """DenseNet121 feature extractor followed by primary and routed capsules."""

    def __init__(
        self,
        *,
        pretrained: bool = False,
        backbone_checkpoint_path: str | Path | None = None,
        freeze_backbone: bool = False,
        feature_h: int = 4,
        feature_w: int = 4,
        primary_caps_dim: int = 8,
        capsule_dim: int = 8,
        num_capsules: int = 256,
        capsule_routing_iters: int = 2,
    ) -> None:
        super().__init__()

        self.backbone = DenseNet121FeatureExtractor(
            pretrained=pretrained,
            checkpoint_path=backbone_checkpoint_path,
            freeze=freeze_backbone,
        )
        flattened_dim = self.backbone.out_channels * feature_h * feature_w
        if flattened_dim % primary_caps_dim != 0:
            raise ValueError(
                f"Flattened feature size {flattened_dim} must be divisible by "
                f"primary_caps_dim={primary_caps_dim}."
            )

        self.pool = nn.AdaptiveAvgPool2d((feature_h, feature_w))
        self.num_primary_caps = flattened_dim // primary_caps_dim
        self.primary_caps_dim = primary_caps_dim
        self.norm = nn.BatchNorm1d(self.num_primary_caps)
        self.capsules = RoutingCapsuleLayer(
            num_in_caps=self.num_primary_caps,
            in_dim=primary_caps_dim,
            num_out_caps=num_capsules,
            out_dim=capsule_dim,
            routing_iters=capsule_routing_iters,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.dim() != 4:
            raise ValueError(f"Expected images [B, C, H, W], got {tuple(images.shape)}")

        feature_map = self.backbone(images)
        batch_size = feature_map.size(0)
        features = self.pool(feature_map)
        primary_capsules = features.flatten(start_dim=1).view(
            batch_size,
            self.num_primary_caps,
            self.primary_caps_dim,
        )
        primary_capsules = self.norm(primary_capsules)
        return self.capsules(primary_capsules)


class PairedDenseCapsFusionHead(nn.Module):
    """Final DigitCaps layer over concatenated whole/detail capsules."""

    def __init__(
        self,
        *,
        num_branch_capsules: int,
        capsule_dim: int,
        num_classes: int,
        digit_caps_dim: int = 16,
        routing_iters: int = 3,
    ) -> None:
        super().__init__()
        self.digit_capsules = RoutingCapsuleLayer(
            num_in_caps=2 * num_branch_capsules,
            in_dim=capsule_dim,
            num_out_caps=num_classes,
            out_dim=digit_caps_dim,
            routing_iters=routing_iters,
        )

    def forward(self, whole_capsules: Tensor, detail_capsules: Tensor) -> tuple[Tensor, Tensor]:
        if whole_capsules.shape != detail_capsules.shape:
            raise ValueError(
                "Whole and detail capsules must have the same shape, got "
                f"{tuple(whole_capsules.shape)} and {tuple(detail_capsules.shape)}."
            )

        fused_capsules = torch.cat([whole_capsules, detail_capsules], dim=1)
        digit_capsules = self.digit_capsules(fused_capsules)
        probabilities = torch.norm(digit_capsules, dim=-1)
        return probabilities, digit_capsules


@dataclass(frozen=True)
class PairedDenseCapsNetConfig:
    """Configuration for the paired whole/detail DenseCaps classifier."""

    num_classes: int
    pretrained: bool = False
    backbone_checkpoint_path: str | Path | None = None
    freeze_backbone: bool = False
    feature_h: int = 4
    feature_w: int = 4
    primary_caps_dim: int = 8
    capsule_dim: int = 8
    digit_caps_dim: int = 16
    num_capsules: int = 256
    capsule_routing_iters: int = 2
    digit_routing_iters: int = 3


class PairedDenseCapsNetClassifier(nn.Module):
    """Classify a whole image and its detailed image with two DenseCaps branches."""

    def __init__(
        self,
        *,
        num_classes: int,
        pretrained: bool = False,
        backbone_checkpoint_path: str | Path | None = None,
        freeze_backbone: bool = False,
        feature_h: int = 4,
        feature_w: int = 4,
        primary_caps_dim: int = 8,
        capsule_dim: int = 8,
        digit_caps_dim: int = 16,
        num_capsules: int = 256,
        capsule_routing_iters: int = 2,
        digit_routing_iters: int = 3,
    ) -> None:
        super().__init__()
        branch_kwargs = {
            "pretrained": pretrained,
            "backbone_checkpoint_path": backbone_checkpoint_path,
            "freeze_backbone": freeze_backbone,
            "feature_h": feature_h,
            "feature_w": feature_w,
            "primary_caps_dim": primary_caps_dim,
            "capsule_dim": capsule_dim,
            "num_capsules": num_capsules,
            "capsule_routing_iters": capsule_routing_iters,
        }
        self.whole_branch = DenseCapsEncoderBranch(**branch_kwargs)
        self.detail_branch = DenseCapsEncoderBranch(**branch_kwargs)
        self.fusion_head = PairedDenseCapsFusionHead(
            num_branch_capsules=num_capsules,
            capsule_dim=capsule_dim,
            num_classes=num_classes,
            digit_caps_dim=digit_caps_dim,
            routing_iters=digit_routing_iters,
        )

    @classmethod
    def from_config(cls, config: PairedDenseCapsNetConfig) -> "PairedDenseCapsNetClassifier":
        return cls(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            backbone_checkpoint_path=config.backbone_checkpoint_path,
            freeze_backbone=config.freeze_backbone,
            feature_h=config.feature_h,
            feature_w=config.feature_w,
            primary_caps_dim=config.primary_caps_dim,
            capsule_dim=config.capsule_dim,
            digit_caps_dim=config.digit_caps_dim,
            num_capsules=config.num_capsules,
            capsule_routing_iters=config.capsule_routing_iters,
            digit_routing_iters=config.digit_routing_iters,
        )

    def forward(self, whole_images: Tensor, detail_images: Tensor) -> Tensor:
        whole_capsules = self.whole_branch(whole_images)
        detail_capsules = self.detail_branch(detail_images)
        probabilities, _ = self.fusion_head(whole_capsules, detail_capsules)
        return probabilities
