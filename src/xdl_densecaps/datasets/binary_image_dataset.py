"""Binary normal/lession image dataset helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Subset


CLASS_NAMES = ("normal", "lession")
NORMAL_DIR_NAMES = {"normal", "nomal"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    """One image path and its binary label."""

    path: Path
    label: int


class BinaryNormalLessionDataset(Dataset[tuple[Tensor, int]]):
    """Map ``normal`` folder images to class 0 and every other folder to class 1."""

    class_names = CLASS_NAMES

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[Image.Image], Tensor] | None = None,
        normal_dir_names: set[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.normal_dir_names = normal_dir_names or NORMAL_DIR_NAMES
        self.samples = find_binary_image_samples(self.root_dir, self.normal_dir_names)

        if not self.samples:
            raise ValueError(f"No supported image files found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label

    def class_counts(self) -> dict[str, int]:
        counts = {class_name: 0 for class_name in self.class_names}
        for sample in self.samples:
            counts[self.class_names[sample.label]] += 1
        return counts


def find_binary_image_samples(root_dir: Path, normal_dir_names: set[str] | None = None) -> list[ImageSample]:
    """Collect images recursively and assign binary labels from the first folder."""

    normal_dir_names = normal_dir_names or NORMAL_DIR_NAMES
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")

    samples: list[ImageSample] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative_parts = path.relative_to(root_dir).parts
        if len(relative_parts) < 2:
            continue

        top_level_folder = relative_parts[0].lower()
        label = 0 if top_level_folder in normal_dir_names else 1
        samples.append(ImageSample(path=path, label=label))

    return samples


def stratified_train_val_split(
    dataset: BinaryNormalLessionDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Subset[tuple[Tensor, int]], Subset[tuple[Tensor, int]]]:
    """Split dataset indices while preserving both binary classes when possible."""

    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in the range [0.0, 1.0).")

    import torch

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in range(len(CLASS_NAMES)):
        indices = [idx for idx, sample in enumerate(dataset.samples) if sample.label == label]
        if not indices:
            continue

        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[idx] for idx in permutation]
        val_count = int(round(len(shuffled) * val_ratio))
        if val_ratio > 0.0 and len(shuffled) > 1:
            val_count = max(1, min(val_count, len(shuffled) - 1))

        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
