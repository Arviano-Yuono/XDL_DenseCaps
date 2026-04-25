"""Strategies for reducing feature maps into lesion heatmaps."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class HeatmapStrategy(ABC):
    """Interface for turning ``[B, C, H, W]`` features into ``[B, H, W]`` maps."""

    @abstractmethod
    def __call__(self, features: Tensor) -> Tensor:
        raise NotImplementedError


class ChannelMeanHeatmap(HeatmapStrategy):
    """Use the channel mean, matching ``torch.mean(data, dim=1)`` in the notebook."""

    def __call__(self, features: Tensor) -> Tensor:
        if features.dim() != 4:
            raise ValueError(f"Expected features [B, C, H, W], got {tuple(features.shape)}")
        return features.mean(dim=1)
