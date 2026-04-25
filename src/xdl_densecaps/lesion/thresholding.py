"""Thresholding strategies for heatmap binarization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor


class ThresholdStrategy(ABC):
    """Interface for converting heatmaps into boolean masks."""

    @abstractmethod
    def __call__(self, heatmaps: Tensor) -> Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class FixedThreshold(ThresholdStrategy):
    """Apply a fixed threshold to the heatmap."""

    threshold: float = 0.02
    greater_than: bool = True

    def __call__(self, heatmaps: Tensor) -> Tensor:
        if heatmaps.dim() not in {2, 3}:
            raise ValueError(f"Expected heatmap [H, W] or [B, H, W], got {tuple(heatmaps.shape)}")

        if self.greater_than:
            return heatmaps > self.threshold
        return heatmaps < self.threshold
