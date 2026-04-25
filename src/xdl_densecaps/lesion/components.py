"""Torch-native connected-component proposal strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


class ConnectedComponentStrategy(ABC):
    """Interface for turning a binary mask into proposal masks."""

    @abstractmethod
    def __call__(self, binary_mask: Tensor) -> Tensor:
        raise NotImplementedError


@dataclass(frozen=True)
class SquareConnectedComponents(ConnectedComponentStrategy):
    """Find 4-connected clusters and replace each cluster with a square mask."""

    aspect_ratio_threshold: float = 4.0
    min_area: int = 1

    def __call__(self, binary_mask: Tensor) -> Tensor:
        return separate_connected_components_as_square_boxes(
            binary_mask=binary_mask,
            aspect_ratio_threshold=self.aspect_ratio_threshold,
            min_area=self.min_area,
        )


def four_connected_neighbors(mask: Tensor) -> Tensor:
    """Return all 4-connected neighbors of the true pixels in ``mask``."""

    neighbors = torch.zeros_like(mask, dtype=torch.bool)
    neighbors[1:, :] |= mask[:-1, :]
    neighbors[:-1, :] |= mask[1:, :]
    neighbors[:, 1:] |= mask[:, :-1]
    neighbors[:, :-1] |= mask[:, 1:]
    return neighbors


def square_bbox_mask(binary_mask: Tensor, y_min: int, x_min: int, y_max: int, x_max: int) -> Tensor:
    """Create a square bool mask that contains the component bounding box."""

    height, width = binary_mask.shape
    bbox_height = y_max - y_min + 1
    bbox_width = x_max - x_min + 1
    side = min(max(bbox_height, bbox_width), height, width)

    y_start = y_min - (side - bbox_height) // 2
    x_start = x_min - (side - bbox_width) // 2
    y_start = max(0, min(y_start, height - side))
    x_start = max(0, min(x_start, width - side))

    mask = torch.zeros_like(binary_mask, dtype=torch.bool)
    mask[y_start : y_start + side, x_start : x_start + side] = True
    return mask


def separate_connected_components_as_square_boxes(
    binary_mask: Tensor,
    aspect_ratio_threshold: float = 4.0,
    min_area: int = 1,
) -> Tensor:
    """Separate a 2D binary mask into square proposal masks.

    Returns:
        Bool tensor with shape ``[num_components, H, W]``.
    """

    if binary_mask.dim() != 2:
        raise ValueError(f"Expected binary_mask [H, W], got {tuple(binary_mask.shape)}")

    binary_mask = binary_mask.bool()
    unvisited = binary_mask.clone()
    square_boxes: list[Tensor] = []

    while bool(unvisited.any()):
        seed = torch.nonzero(unvisited, as_tuple=False)[0]
        y, x = seed[0], seed[1]

        component = torch.zeros_like(binary_mask, dtype=torch.bool)
        frontier = torch.zeros_like(binary_mask, dtype=torch.bool)

        component[y, x] = True
        frontier[y, x] = True
        unvisited[y, x] = False

        while bool(frontier.any()):
            new_frontier = four_connected_neighbors(frontier) & unvisited
            if not bool(new_frontier.any()):
                break

            component |= new_frontier
            unvisited &= ~new_frontier
            frontier = new_frontier

        area = int(component.sum().item())
        if area < min_area:
            continue

        coords = torch.nonzero(component, as_tuple=False)
        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values

        width = float((x_max - x_min + 1).item())
        height = float((y_max - y_min + 1).item())
        aspect_ratio = width / max(height, 1.0)

        if 1.0 / aspect_ratio_threshold < aspect_ratio < aspect_ratio_threshold:
            square_boxes.append(
                square_bbox_mask(
                    binary_mask,
                    y_min=int(y_min.item()),
                    x_min=int(x_min.item()),
                    y_max=int(y_max.item()),
                    x_max=int(x_max.item()),
                )
            )

    if not square_boxes:
        return torch.empty((0, *binary_mask.shape), dtype=torch.bool, device=binary_mask.device)

    return torch.stack(square_boxes, dim=0)
