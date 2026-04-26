"""DenseNet121 backbone with a capsule classification head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xdl_densecaps.models.backbone import DenseNet121FeatureExtractor


def squash(inputs: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """Capsule squash activation."""

    squared_norm = inputs.pow(2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * inputs / torch.sqrt(squared_norm + eps)


class RoutingCapsuleLayer(nn.Module):
    """Fully connected capsule layer with dynamic routing."""

    def __init__(
        self,
        num_in_caps: int,
        in_dim: int,
        num_out_caps: int,
        out_dim: int,
        routing_iters: int = 2,
    ) -> None:
        super().__init__()

        self.num_in_caps = num_in_caps
        self.num_out_caps = num_out_caps
        self.routing_iters = routing_iters
        self.weight = nn.Parameter(
            0.01 * torch.randn(num_in_caps, num_out_caps, in_dim, out_dim)
        )

    def forward(self, capsules: Tensor) -> Tensor:
        if capsules.dim() != 3:
            raise ValueError(f"Expected capsules [B, N, D], got {tuple(capsules.shape)}")

        # [B, N_in, D_in] x [N_in, N_out, D_in, D_out] -> [B, N_in, N_out, D_out]
        predictions = torch.einsum("bid,ijdo->bijo", capsules, self.weight)
        routing_logits = capsules.new_zeros(
            capsules.size(0),
            self.num_in_caps,
            self.num_out_caps,
        )

        for iteration in range(self.routing_iters):
            coupling = F.softmax(routing_logits, dim=2)
            weighted_sum = (coupling.unsqueeze(-1) * predictions).sum(dim=1)
            outputs = squash(weighted_sum, dim=-1)

            if iteration < self.routing_iters - 1:
                agreement = (predictions * outputs.unsqueeze(1)).sum(dim=-1)
                routing_logits = routing_logits + agreement

        return outputs


class DenseNetCapsHead(nn.Module):
    """Capsule head for DenseNet121 feature maps."""

    def __init__(
        self,
        in_channels: int = 1024,
        feature_h: int = 4,
        feature_w: int = 4,
        num_classes: int = 2,
        primary_caps_dim: int = 8,
        capsule_dim: int = 8,
        digit_caps_dim: int = 16,
        num_capsules: int = 256,
        capsule_routing_iters: int = 2,
        digit_routing_iters: int = 3,
    ) -> None:
        super().__init__()

        flattened_dim = in_channels * feature_h * feature_w
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
        self.digit_capsules = RoutingCapsuleLayer(
            num_in_caps=num_capsules,
            in_dim=capsule_dim,
            num_out_caps=num_classes,
            out_dim=digit_caps_dim,
            routing_iters=digit_routing_iters,
        )

    def forward(self, feature_map: Tensor) -> tuple[Tensor, Tensor]:
        if feature_map.dim() != 4:
            raise ValueError(f"Expected feature map [B, C, H, W], got {tuple(feature_map.shape)}")

        batch_size = feature_map.size(0)
        features = self.pool(feature_map)
        primary_capsules = features.flatten(start_dim=1).view(
            batch_size,
            self.num_primary_caps,
            self.primary_caps_dim,
        )
        primary_capsules = self.norm(primary_capsules)

        capsules = self.capsules(primary_capsules)
        digit_capsules = self.digit_capsules(capsules)
        probabilities = torch.norm(digit_capsules, dim=-1)
        return probabilities, digit_capsules


@dataclass(frozen=True)
class DenseNetCapsNetConfig:
    """Configuration for the DenseNet plus CapsNet classifier."""

    num_classes: int = 2
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


class DenseNetCapsNetClassifier(nn.Module):
    """Binary normal/lesion classifier: DenseNet121 features -> CapsNet head."""

    def __init__(
        self,
        num_classes: int = 2,
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

        self.backbone = DenseNet121FeatureExtractor(
            pretrained=pretrained,
            checkpoint_path=backbone_checkpoint_path,
            freeze=freeze_backbone,
        )
        self.caps_head = DenseNetCapsHead(
            in_channels=self.backbone.out_channels,
            feature_h=feature_h,
            feature_w=feature_w,
            num_classes=num_classes,
            primary_caps_dim=primary_caps_dim,
            capsule_dim=capsule_dim,
            digit_caps_dim=digit_caps_dim,
            num_capsules=num_capsules,
            capsule_routing_iters=capsule_routing_iters,
            digit_routing_iters=digit_routing_iters,
        )

    @classmethod
    def from_config(cls, config: DenseNetCapsNetConfig) -> "DenseNetCapsNetClassifier":
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

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone(images)
        probabilities, _ = self.caps_head(features)
        return probabilities


class CapsuleMarginLoss(nn.Module):
    """Margin loss for capsule lengths."""

    def __init__(self, m_plus: float = 0.9, m_minus: float = 0.1, lambda_: float = 0.5) -> None:
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, probabilities: Tensor, labels: Tensor) -> Tensor:
        if labels.dim() != 1:
            labels = labels.view(-1)

        one_hot = F.one_hot(labels.long(), num_classes=probabilities.size(1)).to(probabilities.dtype)
        positive_loss = one_hot * F.relu(self.m_plus - probabilities).pow(2)
        negative_loss = (1.0 - one_hot) * F.relu(probabilities - self.m_minus).pow(2)
        return (positive_loss + self.lambda_ * negative_loss).sum(dim=1).mean()
