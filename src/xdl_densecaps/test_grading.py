"""Test the trained grading stage-1 and stage-2 checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from xdl_densecaps.config import ExperimentConfig, load_config
from xdl_densecaps.evaluation import save_metrics
from xdl_densecaps.train_stage2 import (
    build_stage2_classifier,
    build_stage2_criterion,
    build_stage2_dataset,
    build_stage2_loader,
)
from xdl_densecaps.training import (
    build_classifier,
    build_criterion,
    build_dataset,
    build_loader,
    checkpoint_path,
    format_metrics,
    load_checkpoint,
    load_or_create_split_indices,
    run_epoch,
    run_paired_epoch,
    select_device,
)
from xdl_densecaps.utils import configure_logging, get_logger


DEFAULT_STAGE1_CONFIG = Path("configs/grading-1.yaml")
DEFAULT_STAGE2_CONFIG = Path("configs/grading-2.yaml")
SINGLE_IMAGE_MODEL_NAMES = {"densenet121", "densenet121_capsnet", "densenet_capsnet"}
PAIRED_MODEL_NAMES = {"paired_densenet121_capsnet", "paired_densecaps"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stages = ("stage1", "stage2") if args.stage == "all" else (args.stage,)

    for stage in stages:
        if stage == "stage1":
            config_path = args.stage1_config
            config = _load_with_data_overrides(config_path, root_dir=args.stage1_root)
        else:
            config_path = args.stage2_config
            config = _load_with_data_overrides(
                config_path,
                root_dir=args.stage2_root,
                pair_metadata_path=args.stage2_metadata,
            )

        run_grading_evaluation(config, config_path=config_path, split_name="test")

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test trained grading stage-1 and stage-2 checkpoints.",
    )
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "all"),
        default="all",
        help="Which grading stage to test.",
    )
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=DEFAULT_STAGE1_CONFIG,
        help="Path to the stage-1 grading YAML config.",
    )
    parser.add_argument(
        "--stage2-config",
        type=Path,
        default=DEFAULT_STAGE2_CONFIG,
        help="Path to the stage-2 grading YAML config.",
    )
    parser.add_argument(
        "--stage1-root",
        default=None,
        help="Override stage-1 data.root_dir without editing the YAML file.",
    )
    parser.add_argument(
        "--stage2-root",
        default=None,
        help="Override stage-2 data.root_dir without editing the YAML file.",
    )
    parser.add_argument(
        "--stage2-metadata",
        default=None,
        help="Override stage-2 data.pair_metadata_path without editing the YAML file.",
    )
    return parser.parse_args(argv)


def run_grading_evaluation(
    config: ExperimentConfig,
    *,
    config_path: Path,
    split_name: str = "test",
) -> int:
    model_name = config.model.name.lower()
    if model_name in PAIRED_MODEL_NAMES:
        return run_paired_grading_evaluation(config, config_path=config_path, split_name=split_name)
    if model_name in SINGLE_IMAGE_MODEL_NAMES:
        return run_single_image_grading_evaluation(config, config_path=config_path, split_name=split_name)

    supported = ", ".join(sorted(SINGLE_IMAGE_MODEL_NAMES | PAIRED_MODEL_NAMES))
    raise ValueError(f"Unsupported grading model.name: {config.model.name}. Supported values: {supported}.")


def run_single_image_grading_evaluation(
    config: ExperimentConfig,
    *,
    config_path: Path,
    split_name: str = "test",
) -> int:
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_evaluation_logging(config, output_dir, split_name)
    logger = get_logger("xdl_densecaps.test_grading")

    device = select_device(config.runtime.device)
    dataset, data_root = build_dataset(config)
    split_indices = load_or_create_split_indices(dataset, config, data_root)
    indices = getattr(split_indices, split_name)
    loader = build_loader(config, dataset, indices, split_name, device)

    model = build_classifier(
        config,
        use_pretrained=False,
        use_backbone_checkpoint=False,
    ).to(device)
    best_checkpoint_path = checkpoint_path(config)
    checkpoint = load_checkpoint(best_checkpoint_path, model, device)

    metrics = run_epoch(
        model=model,
        dataloader=loader,
        criterion=build_criterion(config),
        device=device,
        split_name=split_name.capitalize(),
    )

    _log_evaluation(
        logger,
        config_path=config_path,
        log_path=log_path,
        checkpoint_path=best_checkpoint_path,
        checkpoint=checkpoint,
        data_root=data_root,
        split_name=split_name,
        example_count=len(indices),
        metrics=metrics,
    )
    save_metrics(
        output_dir / f"{split_name}_metrics.json",
        split_name=split_name,
        config_path=config_path,
        checkpoint_path=best_checkpoint_path,
        checkpoint_epoch=checkpoint.get("epoch"),
        metrics=metrics,
    )
    return 0


def run_paired_grading_evaluation(
    config: ExperimentConfig,
    *,
    config_path: Path,
    split_name: str = "test",
) -> int:
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _with_portable_pair_metadata(config, output_dir=output_dir)
    log_path = _configure_evaluation_logging(config, output_dir, split_name)
    logger = get_logger("xdl_densecaps.test_grading")

    device = select_device(config.runtime.device)
    dataset, data_root = build_stage2_dataset(config)
    split_indices = load_or_create_split_indices(dataset, config, data_root)
    indices = getattr(split_indices, split_name)
    loader = build_stage2_loader(config, dataset, indices, split_name, device)

    model = build_stage2_classifier(config, num_classes=len(dataset.class_names)).to(device)
    best_checkpoint_path = checkpoint_path(config)
    checkpoint = load_checkpoint(best_checkpoint_path, model, device)

    metrics = run_paired_epoch(
        model=model,
        dataloader=loader,
        criterion=build_stage2_criterion(config),
        device=device,
        split_name=split_name.capitalize(),
    )

    _log_evaluation(
        logger,
        config_path=config_path,
        log_path=log_path,
        checkpoint_path=best_checkpoint_path,
        checkpoint=checkpoint,
        data_root=data_root,
        split_name=split_name,
        example_count=len(indices),
        metrics=metrics,
    )
    save_metrics(
        output_dir / f"{split_name}_metrics.json",
        split_name=split_name,
        config_path=config_path,
        checkpoint_path=best_checkpoint_path,
        checkpoint_epoch=checkpoint.get("epoch"),
        metrics=metrics,
    )
    return 0


def _load_with_data_overrides(
    config_path: Path,
    *,
    root_dir: str | None = None,
    pair_metadata_path: str | None = None,
) -> ExperimentConfig:
    config = load_config(config_path)
    data = config.data
    if root_dir is not None:
        data = replace(data, root_dir=root_dir)
    if pair_metadata_path is not None:
        data = replace(data, pair_metadata_path=pair_metadata_path)
    return replace(config, data=data)


def _with_portable_pair_metadata(config: ExperimentConfig, *, output_dir: Path) -> ExperimentConfig:
    metadata_path = config.data.pair_metadata_path
    if metadata_path is None:
        return config

    source_path = Path(metadata_path)
    if not source_path.exists():
        return config

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    normalized_payload, changed = _normalize_pair_metadata_paths(payload)
    if not changed:
        return config

    portable_path = output_dir / "portable_pair_metadata.json"
    portable_path.parent.mkdir(parents=True, exist_ok=True)
    portable_path.write_text(json.dumps(normalized_payload, indent=2), encoding="utf-8")
    return replace(config, data=replace(config.data, pair_metadata_path=str(portable_path)))


def _normalize_pair_metadata_paths(payload: object) -> tuple[object, bool]:
    if not isinstance(payload, dict):
        return payload, False

    changed = False
    normalized_payload = dict(payload)
    for section in ("normal_records", "records"):
        normalized_records = []
        for record in normalized_payload.get(section, []):
            normalized_record, record_changed = _normalize_pair_metadata_record(record)
            normalized_records.append(normalized_record)
            changed = changed or record_changed
        if section in normalized_payload:
            normalized_payload[section] = normalized_records

    return normalized_payload, changed


def _normalize_pair_metadata_record(record: object) -> tuple[object, bool]:
    if not isinstance(record, dict):
        return record, False

    changed = False
    normalized_record = dict(record)
    for key in ("original_path", "whole_path", "output_path", "detail_path"):
        if key not in normalized_record:
            continue

        normalized_path, path_changed = _normalize_portable_path_value(normalized_record[key])
        normalized_record[key] = normalized_path
        changed = changed or path_changed

    return normalized_record, changed


def _normalize_portable_path_value(path_value: object) -> tuple[object, bool]:
    if not isinstance(path_value, str) or "\\" not in path_value:
        return path_value, False

    original_path = Path(path_value)
    if original_path.exists():
        return path_value, False

    normalized_value = path_value.replace("\\", "/")
    if normalized_value == path_value:
        return path_value, False
    return normalized_value, True


def _configure_evaluation_logging(config: ExperimentConfig, output_dir: Path, split_name: str) -> Path | None:
    log_file = config.runtime.val_log_file if split_name == "val" else config.runtime.test_log_file
    return configure_logging(output_dir, level=config.runtime.log_level, log_file=log_file)


def _log_evaluation(
    logger,
    *,
    config_path: Path,
    log_path: Path | None,
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    data_root: Path,
    split_name: str,
    example_count: int,
    metrics,
) -> None:
    logger.info("Config: %s", config_path)
    if log_path is not None:
        logger.info("Log file: %s", log_path)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Checkpoint epoch: %s", checkpoint.get("epoch"))
    logger.info("Data root: %s", data_root)
    logger.info("Split: %s examples=%s", split_name, example_count)
    logger.info(format_metrics(split_name, metrics))


if __name__ == "__main__":
    raise SystemExit(main())
