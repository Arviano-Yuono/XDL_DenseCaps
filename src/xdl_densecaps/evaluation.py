"""Shared validation and test entrypoint logic."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from xdl_densecaps.config import load_config, parse_config_path
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
    select_device,
)
from xdl_densecaps.utils import configure_logging, get_logger


def run_evaluation_script(
    argv: Sequence[str] | None,
    *,
    split_name: str,
    description: str,
) -> int:
    config_path = parse_config_path(argv, description)
    config = load_config(config_path)
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = config.runtime.val_log_file if split_name == "val" else config.runtime.test_log_file
    log_path = configure_logging(output_dir, level=config.runtime.log_level, log_file=log_file)
    logger = get_logger(f"xdl_densecaps.{split_name}")

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

    criterion = build_criterion(config)
    metrics = run_epoch(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        split_name=split_name.capitalize(),
    )

    logger.info("Config: %s", config_path)
    if log_path is not None:
        logger.info("Log file: %s", log_path)
    logger.info("Checkpoint: %s", best_checkpoint_path)
    logger.info("Checkpoint epoch: %s", checkpoint.get("epoch"))
    logger.info("Data root: %s", data_root)
    logger.info("Split: %s examples=%s", split_name, len(indices))
    logger.info(format_metrics(split_name, metrics))

    save_metrics(
        output_dir / f"{split_name}_metrics.json",
        split_name=split_name,
        config_path=config_path,
        checkpoint_path=best_checkpoint_path,
        checkpoint_epoch=checkpoint.get("epoch"),
        metrics=metrics,
    )
    return 0


def save_metrics(
    path: Path,
    *,
    split_name: str,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_epoch: object,
    metrics,
) -> None:
    payload = {
        "split": split_name,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "metrics": asdict(metrics),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
