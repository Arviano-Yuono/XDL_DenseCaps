"""High-level model that implements the architecture notebook pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor, nn

from xdl_densecaps.lesion.components import ConnectedComponentStrategy, SquareConnectedComponents
from xdl_densecaps.lesion.extraction import (
    CandidateExtractionStrategy,
    LesionCandidate,
    ZoomedMaskCropExtractor,
)
from xdl_densecaps.lesion.heatmap import ChannelMeanHeatmap, HeatmapStrategy
from xdl_densecaps.lesion.thresholding import FixedThreshold, ThresholdStrategy
from xdl_densecaps.models.backbone import DenseNet121FeatureExtractor


@dataclass(frozen=True)
class LesionProposalOutput:
    """Structured output for the lesion proposal model."""

    features: Tensor
    heatmaps: Tensor
    binary_masks: Tensor
    low_res_square_masks: list[Tensor]
    candidates: list[list[LesionCandidate]]

    @property
    def candidate_images(self) -> list[list[Tensor]]:
        return [[candidate.image for candidate in image_candidates] for image_candidates in self.candidates]

    @property
    def candidate_masks(self) -> list[list[Tensor]]:
        return [[candidate.mask for candidate in image_candidates] for image_candidates in self.candidates]


class DenseNetLesionProposalModel(nn.Module):
    """DenseNet-based lesion proposal pipeline.

    Pipeline:
        image -> DenseNet121 trunk -> channel-mean heatmap -> fixed threshold
        -> square 4-connected components -> zoomed crops from the input image.
    """

    def __init__(
        self,
        backbone: DenseNet121FeatureExtractor | None = None,
        heatmap_strategy: HeatmapStrategy | None = None,
        threshold_strategy: ThresholdStrategy | None = None,
        component_strategy: ConnectedComponentStrategy | None = None,
        extraction_strategy: CandidateExtractionStrategy | None = None,
        pretrained: bool = False,
        checkpoint_path: str | Path | None = None,
        freeze_backbone: bool = False,
        binary_threshold: float = 0.02,
        aspect_ratio_threshold: float = 4.0,
        min_component_area: int = 1,
    ) -> None:
        super().__init__()

        self.backbone = backbone or DenseNet121FeatureExtractor(
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            freeze=freeze_backbone,
        )
        self.heatmap_strategy = heatmap_strategy or ChannelMeanHeatmap()
        self.threshold_strategy = threshold_strategy or FixedThreshold(binary_threshold)
        self.component_strategy = component_strategy or SquareConnectedComponents(
            aspect_ratio_threshold=aspect_ratio_threshold,
            min_area=min_component_area,
        )
        self.extraction_strategy = extraction_strategy or ZoomedMaskCropExtractor()

    def forward(self, images: Tensor) -> LesionProposalOutput:
        if images.dim() != 4:
            raise ValueError(f"Expected images [B, C, H, W], got {tuple(images.shape)}")

        features = self.backbone(images)
        heatmaps = self.heatmap_strategy(features)
        binary_masks = self.threshold_strategy(heatmaps)

        all_square_masks: list[Tensor] = []
        all_candidates: list[list[LesionCandidate]] = []

        for image, binary_mask in zip(images, binary_masks):
            square_masks = self.component_strategy(binary_mask)
            all_square_masks.append(square_masks)
            all_candidates.append([self.extraction_strategy(image, mask) for mask in square_masks])

        return LesionProposalOutput(
            features=features,
            heatmaps=heatmaps,
            binary_masks=binary_masks,
            low_res_square_masks=all_square_masks,
            candidates=all_candidates,
        )
