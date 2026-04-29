"""Train the second-stage paired whole/detail DenseCaps classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from torch import nn
from torch.utils.data import DataLoader, Subset

from xdl_densecaps.config import ExperimentConfig, load_config, parse_config_path
from xdl_densecaps.datasets import PairedImageDataset
from xdl_densecaps.models import CapsuleMarginLoss, PairedDenseCapsNetClassifier
from xdl_densecaps.training import (
    build_optimizer,
    build_transform,
    checkpoint_path,
    format_metrics,
    load_or_create_split_indices,
    run_paired_epoch,
    save_checkpoint,
    save_run_metadata,
    select_device,
    set_seed,
)
from xdl_densecaps.utils import configure_logging, get_logger


logger = get_logger("xdl_densecaps.train_stage2")


def main(argv: Sequence[str] | None = None) -> int:
    config_path = parse_config_path(argv, "Train the second-stage paired whole/detail DenseCaps model.")
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
    dataset, data_root = build_stage2_dataset(config)
    split_indices = load_or_create_split_indices(dataset, config, data_root)
    train_loader = build_stage2_loader(config, dataset, split_indices.train, "train", device)
    val_loader = build_stage2_loader(config, dataset, split_indices.val, "val", device)
    model = build_stage2_classifier(config, num_classes=len(dataset.class_names)).to(device)
    criterion = build_stage2_criterion(config)
    optimizer = build_optimizer(config, model)
    best_checkpoint_path = checkpoint_path(config)

    save_run_metadata(output_dir, config, data_root, dataset, split_indices)
    log_startup(config_path, log_path, data_root, dataset.class_counts(), split_indices, dataset.class_names, device)

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = run_paired_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            split_name=f"Stage2 Train {epoch}",
            optimizer=optimizer,
        )
        val_metrics = run_paired_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            split_name=f"Stage2 Val {epoch}",
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
                class_names=dataset.class_names,
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


def build_stage2_dataset(config: ExperimentConfig) -> tuple[PairedImageDataset, Path]:
    dataset = PairedImageDataset(
        config.data.root_dir,
        detail_root_dir=config.data.detail_root_dir,
        metadata_path=config.data.pair_metadata_path,
        class_names=config.data.class_names,
        label_from_parent_dir=config.data.label_from_parent_dir,
    )
    data_root = Path(config.data.pair_metadata_path or config.data.root_dir)
    return dataset, data_root


def build_stage2_loader(
    config: ExperimentConfig,
    dataset: PairedImageDataset,
    indices: list[int],
    split_name: str,
    device,
) -> DataLoader:
    transformed_dataset = dataset.with_transform(build_transform(config, split_name))
    subset = Subset(transformed_dataset, indices)
    pin_memory = config.data.pin_memory and device.type == "cuda"
    return DataLoader(
        subset,
        batch_size=config.data.batch_size,
        shuffle=split_name == "train",
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )


def build_stage2_classifier(config: ExperimentConfig, *, num_classes: int) -> PairedDenseCapsNetClassifier:
    if config.model.name.lower() not in {"paired_densenet121_capsnet", "paired_densecaps"}:
        raise ValueError(
            "Stage-2 training expects model.name to be "
            "'paired_densenet121_capsnet' or 'paired_densecaps'."
        )

    return PairedDenseCapsNetClassifier(
        num_classes=num_classes,
        pretrained=config.model.pretrained,
        backbone_checkpoint_path=config.model.backbone_checkpoint_path,
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


def build_stage2_criterion(config: ExperimentConfig) -> nn.Module:
    return CapsuleMarginLoss(
        m_plus=config.model.margin_m_plus,
        m_minus=config.model.margin_m_minus,
        lambda_=config.model.margin_lambda,
    )


def log_startup(config_path, log_path, data_root, class_counts, split_indices, class_names, device) -> None:
    logger.info("Config: %s", config_path)
    if log_path is not None:
        logger.info("Log file: %s", log_path)
    logger.info("Pair data source: %s", data_root)
    logger.info("Classes: %s", class_names)
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
