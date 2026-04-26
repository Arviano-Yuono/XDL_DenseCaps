"""Train the configured normal/lesion model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from xdl_densecaps.config import load_config, parse_config_path
from xdl_densecaps.training import (
    build_classifier,
    build_criterion,
    build_dataset,
    build_loader,
    build_optimizer,
    checkpoint_path,
    format_metrics,
    load_or_create_split_indices,
    run_epoch,
    save_checkpoint,
    save_run_metadata,
    select_device,
    set_seed,
)
from xdl_densecaps.utils import configure_logging, get_logger


logger = get_logger("xdl_densecaps.train")


def main(argv: Sequence[str] | None = None) -> int:
    config_path = parse_config_path(argv, "Train the configured normal vs lesion model.")
    config = load_config(config_path)
    set_seed(config.data.seed)

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(
        output_dir,
        level=config.runtime.log_level,
        log_file=config.runtime.train_log_file,
    )

    device = select_device(config.runtime.device)
    dataset, data_root = build_dataset(config)
    split_indices = load_or_create_split_indices(dataset, config, data_root)
    train_loader = build_loader(config, dataset, split_indices.train, "train", device)
    val_loader = build_loader(config, dataset, split_indices.val, "val", device)
    model = build_classifier(config).to(device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model)
    best_checkpoint_path = checkpoint_path(config)

    save_run_metadata(output_dir, config, data_root, dataset, split_indices)
    log_startup(config_path, log_path, data_root, dataset.class_counts(), split_indices, device)

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            split_name=f"Train {epoch}",
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            split_name=f"Val {epoch}",
        )

        logger.info(
            "Epoch %s | %s | %s",
            epoch,
            format_metrics("train", train_metrics),
            format_metrics("val", val_metrics),
        )
        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
        else:
            stale_epochs += 1

        if should_stop_early(config.training.early_stopping_patience, stale_epochs):
            logger.info("Early stopping after %s stale epochs.", stale_epochs)
            break

    if best_epoch > 0:
        logger.info("Best checkpoint: %s from epoch %s", best_checkpoint_path, best_epoch)
    else:
        logger.info("No training epochs were run, so no checkpoint was saved.")
    return 0


def log_startup(config_path, log_path, data_root, class_counts, split_indices, device) -> None:
    logger.info("Config: %s", config_path)
    if log_path is not None:
        logger.info("Log file: %s", log_path)
    logger.info("Data root: %s", data_root)
    logger.info("Class counts: %s", class_counts)
    logger.info(
        "Split counts: train=%s val=%s test=%s",
        len(split_indices.train),
        len(split_indices.val),
        len(split_indices.test),
    )
    logger.info("Device: %s", device)


def should_stop_early(patience: int | None, stale_epochs: int) -> bool:
    return patience is not None and patience > 0 and stale_epochs >= patience


if __name__ == "__main__":
    raise SystemExit(main())
