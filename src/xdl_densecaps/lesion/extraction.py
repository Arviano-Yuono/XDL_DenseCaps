"""Candidate extraction from original input images."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class LesionCandidate:
    """A zoomed lesion crop plus the masks used to produce it."""

    image: Tensor
    mask: Tensor
    low_res_mask: Tensor
    bbox_xyxy: BBox


class CandidateExtractionStrategy(ABC):
    """Interface for extracting candidates from an input image and low-res mask."""

    @abstractmethod
    def __call__(self, input_image: Tensor, low_res_mask: Tensor) -> LesionCandidate:
        raise NotImplementedError


@dataclass(frozen=True)
class ZoomedMaskCropExtractor(CandidateExtractionStrategy):
    """Resize the mask, crop the selected region, and zoom it to input size."""

    interpolation_mode: str = "bilinear"

    def __call__(self, input_image: Tensor, low_res_mask: Tensor) -> LesionCandidate:
        image, mask, bbox = extract_zoomed_masked_input_image(
            input_image=input_image,
            low_res_mask=low_res_mask,
            mode=self.interpolation_mode,
        )
        return LesionCandidate(
            image=image,
            mask=mask,
            low_res_mask=low_res_mask.bool(),
            bbox_xyxy=bbox,
        )


def resize_mask_to_input_image(low_res_mask: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Resize a low-resolution bool mask to an input image size."""

    if low_res_mask.dim() != 2:
        raise ValueError(f"Expected low_res_mask [h, w], got {tuple(low_res_mask.shape)}")

    resized_mask = F.interpolate(
        low_res_mask.bool().float().unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode="nearest",
    ).squeeze(0).squeeze(0)
    return resized_mask.bool()


def mask_bounding_box(mask: Tensor) -> BBox | None:
    """Return ``(x_min, y_min, x_max, y_max)`` for a bool mask."""

    coords = torch.nonzero(mask.bool(), as_tuple=False)
    if coords.numel() == 0:
        return None

    y_min, x_min = coords.min(dim=0).values.tolist()
    y_max, x_max = coords.max(dim=0).values.tolist()
    return int(x_min), int(y_min), int(x_max), int(y_max)


def extract_zoomed_masked_input_image(
    input_image: Tensor,
    low_res_mask: Tensor,
    mode: str = "bilinear",
) -> tuple[Tensor, Tensor, BBox]:
    """Apply a resized mask to ``input_image`` and zoom its crop to input size."""

    if input_image.dim() != 3:
        raise ValueError(f"Expected input_image [C, H, W], got {tuple(input_image.shape)}")

    image_mask = resize_mask_to_input_image(low_res_mask, input_image.shape[-2:])
    bbox = mask_bounding_box(image_mask)
    if bbox is None:
        return torch.zeros_like(input_image), image_mask, (0, 0, -1, -1)

    x_min, y_min, x_max, y_max = bbox
    masked_image = input_image * image_mask.unsqueeze(0).to(dtype=input_image.dtype)
    crop = masked_image[:, y_min : y_max + 1, x_min : x_max + 1]

    interpolate_kwargs: dict[str, object] = {
        "size": input_image.shape[-2:],
        "mode": mode,
    }
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = False

    zoomed_image = F.interpolate(crop.unsqueeze(0), **interpolate_kwargs).squeeze(0)
    return zoomed_image, image_mask, bbox
