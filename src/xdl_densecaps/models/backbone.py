"""DenseNet121 feature backbone used by the lesion proposal pipeline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import DenseNet121_Weights, densenet121

from xdl_densecaps.utils.checkpoints import (
    DenseNetFeatureCheckpointAdapter,
    load_checkpoint_state_dict,
)


DENSENET121_TRUNK_LAYERS = (
    "conv0",
    "norm0",
    "relu0",
    "pool0",
    "denseblock1",
    "transition1",
    "denseblock2",
    "transition2",
    "denseblock3",
    "transition3",
    "denseblock4",
)


@dataclass(frozen=True)
class DenseNet121BackboneConfig:
    """Configuration for the truncated DenseNet121 backbone."""

    pretrained: bool = False
    checkpoint_path: str | Path | None = None
    freeze: bool = False


class DenseNet121FeatureExtractor(nn.Module):
    """DenseNet121 trunk ending at ``denseblock4``.

    The output is ReLU-activated to match the notebook step before building the
    binary lesion heatmap.
    """

    out_channels: int

    def __init__(
        self,
        pretrained: bool = False,
        checkpoint_path: str | Path | None = None,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        self.load_result: tuple[list[str], list[str]] | None = None
        if checkpoint_path is not None:
            self.load_result = self._load_feature_checkpoint(backbone, checkpoint_path)

        self.features = nn.Sequential(
            OrderedDict((name, getattr(backbone.features, name)) for name in DENSENET121_TRUNK_LAYERS)
        )
        self.out_channels = backbone.classifier.in_features

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    @classmethod
    def from_config(cls, config: DenseNet121BackboneConfig) -> "DenseNet121FeatureExtractor":
        return cls(
            pretrained=config.pretrained,
            checkpoint_path=config.checkpoint_path,
            freeze=config.freeze,
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        return F.relu(features, inplace=False)

    @staticmethod
    def _load_feature_checkpoint(
        backbone: nn.Module,
        checkpoint_path: str | Path,
    ) -> tuple[list[str], list[str]]:
        state_dict = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
        adapter = DenseNetFeatureCheckpointAdapter(target_prefix="")
        feature_state_dict = adapter.adapt(state_dict)

        if not feature_state_dict:
            raise ValueError(f"No DenseNet feature weights found in checkpoint: {checkpoint_path}")

        missing_keys, unexpected_keys = backbone.features.load_state_dict(
            feature_state_dict,
            strict=False,
        )
        return list(missing_keys), list(unexpected_keys)
