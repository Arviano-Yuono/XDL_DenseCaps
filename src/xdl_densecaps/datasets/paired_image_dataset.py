"""Paired whole/detail image datasets for second-stage training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from xdl_densecaps.datasets.binary_image_dataset import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class PairedImageSample:
    """One whole image, one detailed image, and one class label."""

    whole_path: Path
    detail_path: Path
    label: int
    class_name: str

    @property
    def path(self) -> Path:
        """Compatibility path used by duplicate-aware split helpers."""

        return self.whole_path


class PairedImageDataset(Dataset[tuple[Tensor, Tensor, int]]):
    """Dataset of matched whole-image and detailed-image inputs.

    The dataset can be created from either:
    - a metadata JSON file containing records with ``original_path`` and ``output_path``;
    - two mirrored class-folder roots, where ``detail_root/class/file`` matches
      ``root_dir/class/file``.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        detail_root_dir: str | Path | None = None,
        metadata_path: str | Path | None = None,
        transform: Callable[[Image.Image], Tensor] | None = None,
        whole_transform: Callable[[Image.Image], Tensor] | None = None,
        detail_transform: Callable[[Image.Image], Tensor] | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.detail_root_dir = Path(detail_root_dir) if detail_root_dir is not None else None
        self.metadata_path = Path(metadata_path) if metadata_path is not None else None
        self.whole_transform = whole_transform or transform
        self.detail_transform = detail_transform or transform
        self._configured_class_names = tuple(class_names) if class_names is not None else None

        if self.metadata_path is not None:
            self.samples, self.class_names = find_paired_image_samples_from_metadata(
                self.metadata_path,
                class_names=self._configured_class_names,
            )
        else:
            if self.root_dir is None or self.detail_root_dir is None:
                raise ValueError("PairedImageDataset needs metadata_path or both root_dir and detail_root_dir.")
            self.samples, self.class_names = find_paired_image_samples(
                self.root_dir,
                self.detail_root_dir,
                class_names=self._configured_class_names,
            )

        if not self.samples:
            raise ValueError("No paired whole/detail image samples were found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        sample = self.samples[index]
        whole_image = Image.open(sample.whole_path).convert("RGB")
        detail_image = Image.open(sample.detail_path).convert("RGB")

        if self.whole_transform is not None:
            whole_image = self.whole_transform(whole_image)
        if self.detail_transform is not None:
            detail_image = self.detail_transform(detail_image)

        return whole_image, detail_image, sample.label

    def with_transform(self, transform: Callable[[Image.Image], Tensor]) -> "PairedImageDataset":
        return PairedImageDataset(
            self.root_dir,
            detail_root_dir=self.detail_root_dir,
            metadata_path=self.metadata_path,
            transform=transform,
            class_names=self.class_names,
        )

    def class_counts(self) -> dict[str, int]:
        counts = {class_name: 0 for class_name in self.class_names}
        for sample in self.samples:
            counts[sample.class_name] += 1
        return counts


def find_paired_image_samples(
    root_dir: str | Path,
    detail_root_dir: str | Path,
    *,
    class_names: Sequence[str] | None = None,
) -> tuple[list[PairedImageSample], tuple[str, ...]]:
    """Collect paired samples from mirrored class-folder roots."""

    root = Path(root_dir)
    detail_root = Path(detail_root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Whole-image root does not exist: {root}")
    if not detail_root.exists():
        raise FileNotFoundError(f"Detail-image root does not exist: {detail_root}")

    discovered_classes = tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))
    resolved_class_names = tuple(class_names) if class_names is not None else discovered_classes
    class_to_label = {class_name: label for label, class_name in enumerate(resolved_class_names)}

    samples: list[PairedImageSample] = []
    for whole_path in sorted(root.rglob("*")):
        if not whole_path.is_file() or whole_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_path = whole_path.relative_to(root)
        if len(relative_path.parts) < 2:
            continue

        class_name = relative_path.parts[0]
        if class_name not in class_to_label:
            continue

        detail_path = _matching_detail_path(detail_root, relative_path)
        if detail_path is None:
            continue

        samples.append(
            PairedImageSample(
                whole_path=whole_path,
                detail_path=detail_path,
                label=class_to_label[class_name],
                class_name=class_name,
            )
        )

    return samples, resolved_class_names


def find_paired_image_samples_from_metadata(
    metadata_path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
) -> tuple[list[PairedImageSample], tuple[str, ...]]:
    """Collect paired samples from filter-region metadata records."""

    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Pair metadata must be a JSON object: {path}")

    raw_records = [
        *payload.get("normal_records", []),
        *payload.get("records", []),
    ]
    if not raw_records:
        raise ValueError(f"Pair metadata has no records: {path}")

    resolved_class_names = _metadata_class_names(payload, raw_records, class_names)
    class_to_label = {class_name: label for label, class_name in enumerate(resolved_class_names)}
    samples: list[PairedImageSample] = []

    for record in raw_records:
        if not isinstance(record, dict):
            continue

        whole_path_value = record.get("original_path") or record.get("whole_path")
        detail_path_value = record.get("output_path") or record.get("detail_path")
        if whole_path_value is None or detail_path_value is None:
            raise ValueError("Each pair metadata record needs original_path/whole_path and output_path/detail_path.")

        class_name = record.get("true_class") or record.get("class_name")
        label = record.get("true_label", record.get("label"))
        if class_name is None:
            if label is None:
                raise ValueError("Each pair metadata record needs true_class/class_name or true_label/label.")
            class_name = resolved_class_names[int(label)]

        class_name = str(class_name)
        if class_name not in class_to_label:
            raise ValueError(f"Record class {class_name!r} is not in class_names={resolved_class_names!r}.")

        samples.append(
            PairedImageSample(
                whole_path=_resolve_metadata_path(whole_path_value, metadata_dir=path.parent),
                detail_path=_resolve_metadata_path(detail_path_value, metadata_dir=path.parent),
                label=class_to_label[class_name],
                class_name=class_name,
            )
        )

    return samples, resolved_class_names


def _metadata_class_names(
    payload: dict[str, object],
    records: list[object],
    class_names: Sequence[str] | None,
) -> tuple[str, ...]:
    if class_names is not None:
        return tuple(str(class_name) for class_name in class_names)

    settings = payload.get("settings")
    if isinstance(settings, dict) and isinstance(settings.get("class_names"), list):
        return tuple(str(class_name) for class_name in settings["class_names"])

    discovered = sorted(
        {
            str(record.get("true_class") or record.get("class_name"))
            for record in records
            if isinstance(record, dict) and (record.get("true_class") or record.get("class_name"))
        }
    )
    if not discovered:
        raise ValueError("Could not infer class names from pair metadata.")
    return tuple(discovered)


def _resolve_metadata_path(path_value: object, *, metadata_dir: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() or path.exists():
        return path

    metadata_relative_path = metadata_dir / path
    if metadata_relative_path.exists():
        return metadata_relative_path
    return path


def _matching_detail_path(detail_root: Path, relative_path: Path) -> Path | None:
    exact_path = detail_root / relative_path
    if exact_path.exists():
        return exact_path

    stem_path = exact_path.with_suffix("")
    for extension in sorted(IMAGE_EXTENSIONS):
        candidate = stem_path.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None
