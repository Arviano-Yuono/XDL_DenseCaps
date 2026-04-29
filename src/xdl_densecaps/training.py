"""Shared training and evaluation helpers."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from xdl_densecaps.config import ExperimentConfig
from xdl_densecaps.datasets import CLASS_NAMES, BinaryNormalLesionDataset
from xdl_densecaps.models import CapsuleMarginLoss, DenseNet121Classifier, DenseNetCapsNetClassifier


SPLIT_FILENAME = "splits.json"
SPLIT_STRATEGY = "sha256_grouped_v1"


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated metrics for one epoch."""

    loss: float
    accuracy: float
    examples: int


@dataclass(frozen=True)
class SplitIndices:
    """Dataset indices for train, validation, and test splits."""

    train: list[int]
    val: list[int]
    test: list[int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_data_root(requested_root: str | Path) -> Path:
    """Resolve the dataset root while tolerating the repo's older data/raw layout."""

    requested_path = Path(requested_root)
    if requested_path.exists():
        return requested_path

    fallback = Path("data/raw")
    if requested_path == Path("datasets/raw") and fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Dataset root does not exist: {requested_path}. "
        "Set data.root_dir in the YAML config to the folder containing normal/ and lesion folders."
    )


def build_classifier(
    config: ExperimentConfig,
    *,
    use_pretrained: bool | None = None,
    use_backbone_checkpoint: bool = True,
) -> nn.Module:
    """Build the configured classifier."""

    model_name = config.model.name.lower()
    pretrained = config.model.pretrained if use_pretrained is None else use_pretrained

    if model_name == "densenet121":
        return DenseNet121Classifier(
            num_classes=len(CLASS_NAMES),
            pretrained=pretrained,
            dropout=config.model.dropout,
            freeze_features=config.model.freeze_backbone,
        )

    if model_name in {"densenet121_capsnet", "densenet_capsnet"}:
        backbone_checkpoint_path = (
            config.model.backbone_checkpoint_path if use_backbone_checkpoint else None
        )
        return DenseNetCapsNetClassifier(
            num_classes=len(CLASS_NAMES),
            pretrained=pretrained,
            backbone_checkpoint_path=backbone_checkpoint_path,
            freeze_backbone=config.model.freeze_backbone,
            feature_h=config.model.feature_h,
            feature_w=config.model.feature_w,
            primary_caps_dim=config.model.primary_caps_dim,
            capsule_dim=config.model.capsule_dim,
            digit_caps_dim=config.model.digit_caps_dim,
            num_capsules=config.model.num_capsules,
            capsule_routing_iters=config.model.capsule_routing_iters,
            digit_routing_iters=config.model.digit_routing_iters,
        )

    supported = "densenet121, densenet121_capsnet"
    raise ValueError(f"Unsupported model.name: {config.model.name}. Supported values: {supported}.")


def build_criterion(config: ExperimentConfig) -> nn.Module:
    """Build the loss function that matches the configured model."""

    model_name = config.model.name.lower()
    if model_name in {"densenet121_capsnet", "densenet_capsnet"}:
        return CapsuleMarginLoss(
            m_plus=config.model.margin_m_plus,
            m_minus=config.model.margin_m_minus,
            lambda_=config.model.margin_lambda,
        )

    return nn.CrossEntropyLoss()


def build_optimizer(config: ExperimentConfig, model: nn.Module) -> AdamW:
    return AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def build_dataset(config: ExperimentConfig, transform=None) -> tuple[BinaryNormalLesionDataset, Path]:
    data_root = resolve_data_root(config.data.root_dir)
    return BinaryNormalLesionDataset(data_root, transform=transform), data_root


def build_loader(
    config: ExperimentConfig,
    dataset: BinaryNormalLesionDataset,
    indices: list[int],
    split_name: str,
    device: torch.device,
) -> DataLoader[tuple[Tensor, Tensor]]:
    transformed_dataset = BinaryNormalLesionDataset(
        dataset.root_dir,
        transform=build_transform(config, split_name),
    )
    subset = Subset(transformed_dataset, indices)
    pin_memory = config.data.pin_memory and device.type == "cuda"

    return DataLoader(
        subset,
        batch_size=config.data.batch_size,
        shuffle=split_name == "train",
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )


def build_transform(config: ExperimentConfig, split_name: str):
    transform_steps: list[object] = [transforms.Resize((config.data.image_size, config.data.image_size))]
    if split_name == "train" and config.data.augment:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
            ]
        )
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


def load_or_create_split_indices(
    dataset: BinaryNormalLesionDataset,
    config: ExperimentConfig,
    data_root: Path,
) -> SplitIndices:
    split_path = Path(config.data.split_dir) / SPLIT_FILENAME
    expected_metadata = _split_metadata(dataset, config, data_root)

    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as file:
            split_payload = json.load(file)
        if split_payload.get("metadata") == expected_metadata:
            return SplitIndices(
                train=list(split_payload["indices"]["train"]),
                val=list(split_payload["indices"]["val"]),
                test=list(split_payload["indices"]["test"]),
            )

    split_indices = create_split_indices(
        dataset,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.seed,
    )
    save_split_indices(split_path, split_indices, expected_metadata)
    return split_indices


def create_split_indices(
    dataset,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitIndices:
    """Create duplicate-aware train/val/test indices.

    Exact duplicate files are grouped by SHA-256 and assigned to the same split
    so the same image content cannot appear in train, validation, and test.
    """

    if val_ratio < 0.0 or test_ratio < 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.0.")

    generator = torch.Generator().manual_seed(seed)
    groups = _duplicate_groups(dataset)
    permutation = torch.randperm(len(groups), generator=generator).tolist()
    shuffled_groups = [groups[index] for index in permutation]
    shuffled_groups.sort(key=len, reverse=True)

    targets = _split_targets(dataset, val_ratio, test_ratio)
    split_indices = {"train": [], "val": [], "test": []}
    num_classes = len(dataset_class_names(dataset))
    split_label_counts = {
        split_name: {label: 0 for label in range(num_classes)}
        for split_name in split_indices
    }

    for group in shuffled_groups:
        group_label_counts = Counter(dataset.samples[index].label for index in group)
        split_name = _choose_split_for_group(group_label_counts, split_label_counts, targets)

        split_indices[split_name].extend(group)
        for label, count in group_label_counts.items():
            split_label_counts[split_name][label] += count

    return SplitIndices(
        train=split_indices["train"],
        val=split_indices["val"],
        test=split_indices["test"],
    )


def save_split_indices(path: Path, split_indices: SplitIndices, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "indices": {
            "train": split_indices.train,
            "val": split_indices.val,
            "test": split_indices.test,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(dataloader, desc=split_name, leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_training):
            scores = model(images)
            loss = criterion(scores, targets)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (scores.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

        progress.set_postfix(
            loss=total_loss / max(total_examples, 1),
            acc=total_correct / max(total_examples, 1),
        )

    if total_examples == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0, examples=0)

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def run_paired_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(dataloader, desc=split_name, leave=False)
    for whole_images, detail_images, targets in progress:
        whole_images = whole_images.to(device)
        detail_images = detail_images.to(device)
        targets = targets.to(device)

        with torch.set_grad_enabled(is_training):
            scores = model(whole_images, detail_images)
            loss = criterion(scores, targets)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = whole_images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (scores.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

        progress.set_postfix(
            loss=total_loss / max(total_examples, 1),
            acc=total_correct / max(total_examples, 1),
        )

    if total_examples == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0, examples=0)

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: ExperimentConfig,
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
    class_names: Sequence[str] = CLASS_NAMES,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": list(class_names),
            "config": asdict(config),
            "train_metrics": asdict(train_metrics),
            "val_metrics": asdict(val_metrics),
        },
        path,
    )


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_run_metadata(
    output_dir: Path,
    config: ExperimentConfig,
    data_root: Path,
    dataset: BinaryNormalLesionDataset,
    split_indices: SplitIndices,
) -> None:
    class_names = dataset_class_names(dataset)
    metadata = {
        "data_root": str(data_root),
        "class_names": list(class_names),
        "class_counts": dataset.class_counts(),
        "split_counts": {
            "train": len(split_indices.train),
            "val": len(split_indices.val),
            "test": len(split_indices.test),
        },
        "config": asdict(config),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def format_metrics(split_name: str, metrics: EpochMetrics) -> str:
    return (
        f"{split_name}: "
        f"loss={metrics.loss:.4f} acc={metrics.accuracy:.4f} examples={metrics.examples}"
    )


def checkpoint_path(config: ExperimentConfig) -> Path:
    return Path(config.runtime.output_dir) / config.runtime.checkpoint_name


def _split_count(total_count: int, ratio: float) -> int:
    split_count = int(round(total_count * ratio))
    if ratio > 0.0 and total_count > 1:
        split_count = max(1, split_count)
    return split_count


def _duplicate_groups(dataset: BinaryNormalLesionDataset) -> list[list[int]]:
    groups_by_hash: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        groups_by_hash[_file_sha256(sample.path)].append(index)
    return list(groups_by_hash.values())


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_targets(
    dataset,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, dict[int, int]]:
    num_classes = len(dataset_class_names(dataset))
    targets = {
        "train": {label: 0 for label in range(num_classes)},
        "val": {label: 0 for label in range(num_classes)},
        "test": {label: 0 for label in range(num_classes)},
    }

    for label in range(num_classes):
        total_count = sum(1 for sample in dataset.samples if sample.label == label)
        test_count = _split_count(total_count, test_ratio)
        val_count = _split_count(total_count, val_ratio)

        while test_count + val_count >= total_count and test_count + val_count > 0:
            if val_count >= test_count and val_count > 0:
                val_count -= 1
            elif test_count > 0:
                test_count -= 1

        targets["test"][label] = test_count
        targets["val"][label] = val_count
        targets["train"][label] = total_count - test_count - val_count

    return targets


def _choose_split_for_group(
    group_label_counts: Counter[int],
    split_label_counts: dict[str, dict[int, int]],
    targets: dict[str, dict[int, int]],
) -> str:
    best_split = "train"
    best_score = -1

    for split_name in ("test", "val", "train"):
        score = 0
        for label, group_count in group_label_counts.items():
            remaining = targets[split_name][label] - split_label_counts[split_name][label]
            score += min(group_count, max(remaining, 0))

        if score > best_score:
            best_split = split_name
            best_score = score

    return best_split


def _split_metadata(
    dataset: BinaryNormalLesionDataset,
    config: ExperimentConfig,
    data_root: Path,
) -> dict[str, object]:
    metadata = {
        "split_strategy": SPLIT_STRATEGY,
        "data_root": str(data_root.resolve()),
        "sample_count": len(dataset),
        "class_names": list(dataset_class_names(dataset)),
        "class_counts": dataset.class_counts(),
        "val_ratio": config.data.val_ratio,
        "test_ratio": config.data.test_ratio,
        "seed": config.data.seed,
    }
    if config.data.detail_root_dir is not None:
        metadata["detail_root_dir"] = str(Path(config.data.detail_root_dir).resolve())
    if config.data.pair_metadata_path is not None:
        metadata["pair_metadata_path"] = str(Path(config.data.pair_metadata_path).resolve())
    return metadata


def dataset_class_names(dataset) -> tuple[str, ...]:
    return tuple(getattr(dataset, "class_names", CLASS_NAMES))
