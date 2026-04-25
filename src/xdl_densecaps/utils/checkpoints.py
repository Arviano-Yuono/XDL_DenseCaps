"""Checkpoint helpers used by the model modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def load_checkpoint_state_dict(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Load a PyTorch checkpoint and return its model state dict.

    Supports both raw state dict checkpoints and dictionaries containing a
    ``model_state_dict`` entry. ``module.`` prefixes from DataParallel are
    stripped because they are an implementation detail of the training run.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}. "
            "If you want to train from scratch, set model.backbone_checkpoint_path to null."
        )

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    return unwrap_state_dict(checkpoint)


def unwrap_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    """Normalize common checkpoint formats into a tensor state dict."""

    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a state dict or contain 'model_state_dict'.")

    state_dict: dict[str, Tensor] = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            state_dict[str(key).removeprefix("module.")] = value
    return state_dict


@dataclass(frozen=True)
class DenseNetFeatureCheckpointAdapter:
    """Adapt DenseNet feature keys from different training wrappers.

    The architecture notebook loaded weights trained with a custom class whose
    feature keys looked like ``densenet_model.features.conv0.weight``. This
    adapter also accepts torchvision-style ``features.*`` keys and full model
    keys such as ``backbone.features.*``.
    """

    target_prefix: str = ""
    source_prefixes: tuple[str, ...] = (
        "densenet_model.features.",
        "model.features.",
        "backbone.features.",
        "features.",
    )

    def adapt(self, state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
        adapted: dict[str, Tensor] = {}

        for key, value in state_dict.items():
            for source_prefix in self.source_prefixes:
                if key.startswith(source_prefix):
                    feature_key = key.removeprefix(source_prefix)
                    adapted[f"{self.target_prefix}{feature_key}"] = value
                    break

        return adapted
