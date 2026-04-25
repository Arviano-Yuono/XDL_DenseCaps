"""DenseNet classifier models."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


@dataclass(frozen=True)
class DenseNetClassifierConfig:
    """Configuration for a DenseNet121 classifier."""

    num_classes: int = 2
    pretrained: bool = False
    dropout: float = 0.25
    freeze_features: bool = False


class DenseNet121Classifier(nn.Module):
    """DenseNet121 classifier with a small replaceable classification head."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.25,
        freeze_features: bool = False,
    ) -> None:
        super().__init__()

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = densenet121(weights=weights)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        if freeze_features:
            for parameter in self.model.features.parameters():
                parameter.requires_grad = False

    @classmethod
    def from_config(cls, config: DenseNetClassifierConfig) -> "DenseNet121Classifier":
        return cls(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
            freeze_features=config.freeze_features,
        )

    def forward(self, images):
        return self.model(images)
