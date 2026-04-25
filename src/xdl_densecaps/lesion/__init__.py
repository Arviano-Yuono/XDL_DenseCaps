"""Lesion proposal strategies."""

from xdl_densecaps.lesion.components import (
    ConnectedComponentStrategy,
    SquareConnectedComponents,
    four_connected_neighbors,
    separate_connected_components_as_square_boxes,
    square_bbox_mask,
)
from xdl_densecaps.lesion.extraction import (
    CandidateExtractionStrategy,
    LesionCandidate,
    ZoomedMaskCropExtractor,
    extract_zoomed_masked_input_image,
    mask_bounding_box,
    resize_mask_to_input_image,
)
from xdl_densecaps.lesion.heatmap import ChannelMeanHeatmap, HeatmapStrategy
from xdl_densecaps.lesion.thresholding import FixedThreshold, ThresholdStrategy

__all__ = [
    "CandidateExtractionStrategy",
    "ChannelMeanHeatmap",
    "ConnectedComponentStrategy",
    "FixedThreshold",
    "HeatmapStrategy",
    "LesionCandidate",
    "SquareConnectedComponents",
    "ThresholdStrategy",
    "ZoomedMaskCropExtractor",
    "extract_zoomed_masked_input_image",
    "four_connected_neighbors",
    "mask_bounding_box",
    "resize_mask_to_input_image",
    "separate_connected_components_as_square_boxes",
    "square_bbox_mask",
]
