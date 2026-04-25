"""YAML experiment configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence, TypeVar

import yaml


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
T = TypeVar("T")


@dataclass(frozen=True)
class DataConfig:
    """Dataset and loader settings."""

    root_dir: str = "datasets/raw"
    split_dir: str = "data/splits/baseline"
    image_size: int = 128
    batch_size: int = 16
    num_workers: int = 0
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    seed: int = 42
    augment: bool = True
    pin_memory: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Model settings."""

    name: str = "densenet121_capsnet"
    pretrained: bool = False
    freeze_backbone: bool = False
    dropout: float = 0.25
    backbone_checkpoint_path: str | None = None
    feature_h: int = 4
    feature_w: int = 4
    primary_caps_dim: int = 8
    capsule_dim: int = 8
    digit_caps_dim: int = 16
    num_capsules: int = 256
    capsule_routing_iters: int = 2
    digit_routing_iters: int = 3
    margin_m_plus: float = 0.9
    margin_m_minus: float = 0.1
    margin_lambda: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings."""

    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime and artifact settings."""

    output_dir: str = "artifacts/normal_lession_densenet121"
    device: str = "auto"
    checkpoint_name: str = "best.pt"
    log_level: str = "INFO"
    train_log_file: str = "train.log"
    val_log_file: str = "val.log"
    test_log_file: str = "test.log"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config loaded from YAML."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """Load an experiment config from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    return ExperimentConfig(
        data=_build_dataclass(DataConfig, raw_config.get("data", {})),
        model=_build_dataclass(ModelConfig, raw_config.get("model", {})),
        training=_build_dataclass(TrainingConfig, raw_config.get("training", {})),
        runtime=_build_dataclass(RuntimeConfig, raw_config.get("runtime", {})),
    )


def parse_config_path(argv: Sequence[str] | None, description: str) -> Path:
    """Parse the only CLI input: which YAML config to use."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the YAML config file.")
    args = parser.parse_args(argv)
    return Path(args.config)


def _build_dataclass(cls: type[T], values: Any) -> T:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} values must be a mapping.")

    allowed_fields = {field.name for field in fields(cls)}
    filtered_values = {key: value for key, value in values.items() if key in allowed_fields}
    return cls(**filtered_values)
