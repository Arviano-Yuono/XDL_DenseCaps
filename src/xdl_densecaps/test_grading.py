"""Test the trained grading stage-1 and stage-2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import torch
from xdl_densecaps.config import ExperimentConfig, load_config
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
    splits = ("train", "val", "test") if args.split == "all" else (args.split,)

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

        for split_name in splits:
            run_grading_evaluation(config, config_path=config_path, split_name=split_name)

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
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="Which split to evaluate with the best checkpoint.",
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
    loader = build_loader(config, dataset, indices, _loader_eval_split_name(split_name), device)

    model = build_classifier(
        config,
        use_pretrained=False,
        use_backbone_checkpoint=False,
    ).to(device)
    best_checkpoint_path = checkpoint_path(config)
    checkpoint = load_checkpoint(best_checkpoint_path, model, device)

    metrics, report = evaluate_single_image_model(
        model=model,
        dataloader=loader,
        criterion=build_criterion(config),
        device=device,
        split_name=split_name.capitalize(),
        class_names=dataset.class_names,
    )
    report = _with_metric_summary(report, metrics)
    confusion_matrix_path = output_dir / f"{split_name}_confusion_matrix.png"
    save_confusion_matrix_image(report, confusion_matrix_path, title=f"{split_name.capitalize()} confusion matrix")
    report = {**report, "confusion_matrix_image": str(confusion_matrix_path)}

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
        report=report,
    )
    save_grading_metrics(
        output_dir / f"{split_name}_metrics.json",
        split_name=split_name,
        config_path=config_path,
        checkpoint_path=best_checkpoint_path,
        checkpoint_epoch=checkpoint.get("epoch"),
        metrics=metrics,
        report=report,
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
    loader = build_stage2_loader(config, dataset, indices, _loader_eval_split_name(split_name), device)

    model = build_stage2_classifier(config, num_classes=len(dataset.class_names)).to(device)
    best_checkpoint_path = checkpoint_path(config)
    checkpoint = load_checkpoint(best_checkpoint_path, model, device)

    metrics, report = evaluate_paired_model(
        model=model,
        dataloader=loader,
        criterion=build_stage2_criterion(config),
        device=device,
        split_name=split_name.capitalize(),
        class_names=dataset.class_names,
    )
    report = _with_metric_summary(report, metrics)
    confusion_matrix_path = output_dir / f"{split_name}_confusion_matrix.png"
    save_confusion_matrix_image(report, confusion_matrix_path, title=f"{split_name.capitalize()} confusion matrix")
    report = {**report, "confusion_matrix_image": str(confusion_matrix_path)}

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
        report=report,
    )
    save_grading_metrics(
        output_dir / f"{split_name}_metrics.json",
        split_name=split_name,
        config_path=config_path,
        checkpoint_path=best_checkpoint_path,
        checkpoint_epoch=checkpoint.get("epoch"),
        metrics=metrics,
        report=report,
    )
    return 0


def evaluate_single_image_model(
    *,
    model,
    dataloader,
    criterion,
    device: torch.device,
    split_name: str,
    class_names: Sequence[str],
):
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    progress = _progress(dataloader, split_name)
    with torch.no_grad():
        for images, batch_targets in progress:
            images = images.to(device)
            batch_targets = batch_targets.to(device)
            scores = model(images)
            loss = criterion(scores, batch_targets)
            batch_predictions = scores.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            targets.extend(int(label) for label in batch_targets.detach().cpu().tolist())
            predictions.extend(int(label) for label in batch_predictions.detach().cpu().tolist())
            _set_progress_metrics(progress, total_loss, targets, predictions)

    return _build_evaluation_outputs(
        total_loss=total_loss,
        targets=targets,
        predictions=predictions,
        class_names=class_names,
    )


def evaluate_paired_model(
    *,
    model,
    dataloader,
    criterion,
    device: torch.device,
    split_name: str,
    class_names: Sequence[str],
):
    model.eval()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    progress = _progress(dataloader, split_name)
    with torch.no_grad():
        for whole_images, detail_images, batch_targets in progress:
            whole_images = whole_images.to(device)
            detail_images = detail_images.to(device)
            batch_targets = batch_targets.to(device)
            scores = model(whole_images, detail_images)
            loss = criterion(scores, batch_targets)
            batch_predictions = scores.argmax(dim=1)

            batch_size = whole_images.size(0)
            total_loss += loss.item() * batch_size
            targets.extend(int(label) for label in batch_targets.detach().cpu().tolist())
            predictions.extend(int(label) for label in batch_predictions.detach().cpu().tolist())
            _set_progress_metrics(progress, total_loss, targets, predictions)

    return _build_evaluation_outputs(
        total_loss=total_loss,
        targets=targets,
        predictions=predictions,
        class_names=class_names,
    )


def save_grading_metrics(
    path: Path,
    *,
    split_name: str,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_epoch: object,
    metrics,
    report: dict[str, object],
) -> None:
    payload = {
        "split": split_name,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "metrics": {
            **asdict(metrics),
            **report,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_confusion_matrix_image(report: dict[str, object], path: Path, *, title: str) -> None:
    matplotlib_config_dir = path.parent / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    confusion_matrix = report["confusion_matrix"]
    if not isinstance(confusion_matrix, dict):
        raise ValueError("classification report does not contain a confusion matrix.")

    labels = [str(label) for label in confusion_matrix["labels"]]
    matrix = confusion_matrix["matrix"]
    matrix_size = max(len(labels), 1)
    heatmap_size = max(5.0, 1.2 * matrix_size)
    table_width = max(5.5, 0.9 * matrix_size + 3.0)

    fig, (matrix_ax, metrics_ax) = plt.subplots(
        1,
        2,
        figsize=(heatmap_size + table_width, heatmap_size),
        gridspec_kw={"width_ratios": [heatmap_size, table_width]},
    )
    image = matrix_ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=matrix_ax, fraction=0.046, pad=0.04)

    matrix_ax.set_title(title)
    matrix_ax.set_xlabel("Predicted label")
    matrix_ax.set_ylabel("True label")
    matrix_ax.set_xticks(range(len(labels)))
    matrix_ax.set_yticks(range(len(labels)))
    matrix_ax.set_xticklabels(labels, rotation=45, ha="right")
    matrix_ax.set_yticklabels(labels)

    max_count = max((max(row) for row in matrix), default=0)
    threshold = max_count / 2.0
    for row_index, row in enumerate(matrix):
        for col_index, count in enumerate(row):
            color = "white" if count > threshold else "black"
            matrix_ax.text(col_index, row_index, str(count), ha="center", va="center", color=color)

    _draw_metrics_table(metrics_ax, report)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _with_metric_summary(report: dict[str, object], metrics) -> dict[str, object]:
    return {
        **report,
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "examples": metrics.examples,
    }


def _draw_metrics_table(ax, report: dict[str, object]) -> None:
    ax.axis("off")
    ax.set_title("Metrics", pad=12)

    summary_rows = [
        ["Examples", str(report.get("examples", "-"))],
        ["Loss", _format_metric(report.get("loss"))],
        ["Accuracy", _format_metric(report.get("accuracy"))],
        ["Macro precision", _format_metric(report["macro_precision"])],
        ["Macro sensitivity", _format_metric(report["macro_sensitivity"])],
        ["Macro specificity", _format_metric(report["macro_specificity"])],
        ["Macro F1", _format_metric(report["macro_f1"])],
        ["Weighted F1", _format_metric(report["weighted_f1"])],
    ]
    summary_table = ax.table(
        cellText=summary_rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.72, 1.0, 0.24],
    )
    _style_table(summary_table, font_size=9)

    class_rows = [
        [
            item["class_name"],
            item["support"],
            _format_metric(item["precision"]),
            _format_metric(item["sensitivity"]),
            _format_metric(item["specificity"]),
            _format_metric(item["f1"]),
        ]
        for item in report["per_class"]
    ]
    class_table = ax.table(
        cellText=class_rows,
        colLabels=["Class", "N", "Prec", "Sens", "Spec", "F1"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.05, 1.0, 0.58],
    )
    _style_table(class_table, font_size=8)


def _style_table(table, *, font_size: int) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#eaeef2")
            cell.set_text_props(weight="bold")


def _format_metric(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "-"


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
    if split_name == "train":
        log_file = "train_eval.log"
    elif split_name == "val":
        log_file = config.runtime.val_log_file
    else:
        log_file = config.runtime.test_log_file
    return configure_logging(output_dir, level=config.runtime.log_level, log_file=log_file)


def _loader_eval_split_name(split_name: str) -> str:
    if split_name == "train":
        return "val"
    return split_name


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
    report: dict[str, object],
) -> None:
    logger.info("Config: %s", config_path)
    if log_path is not None:
        logger.info("Log file: %s", log_path)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Checkpoint epoch: %s", checkpoint.get("epoch"))
    logger.info("Data root: %s", data_root)
    logger.info("Split: %s examples=%s", split_name, example_count)
    logger.info(format_metrics(split_name, metrics))
    logger.info(
        "macro: precision=%.4f sensitivity=%.4f specificity=%.4f f1=%.4f",
        report["macro_precision"],
        report["macro_sensitivity"],
        report["macro_specificity"],
        report["macro_f1"],
    )
    logger.info("Confusion matrix image: %s", report["confusion_matrix_image"])
    logger.info("Metrics table:\n%s", _format_metrics_table(split_name, metrics, report))


def _build_evaluation_outputs(
    *,
    total_loss: float,
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
):
    from xdl_densecaps.training import EpochMetrics

    examples = len(targets)
    accuracy = _safe_div(sum(1 for true, pred in zip(targets, predictions) if true == pred), examples)
    loss = _safe_div(total_loss, examples)
    metrics = EpochMetrics(loss=loss, accuracy=accuracy, examples=examples)
    report = _classification_report(targets=targets, predictions=predictions, class_names=class_names)
    return metrics, report


def _classification_report(
    *,
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, object]:
    num_classes = len(class_names)
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions):
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            matrix[target][prediction] += 1

    total = sum(sum(row) for row in matrix)
    per_class = []
    for label, class_name in enumerate(class_names):
        true_positive = matrix[label][label]
        false_negative = sum(matrix[label]) - true_positive
        false_positive = sum(row[label] for row in matrix) - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        precision = _safe_div(true_positive, true_positive + false_positive)
        sensitivity = _safe_div(true_positive, true_positive + false_negative)
        specificity = _safe_div(true_negative, true_negative + false_positive)
        f1 = _safe_div(2.0 * precision * sensitivity, precision + sensitivity)
        support = true_positive + false_negative

        per_class.append(
            {
                "label": label,
                "class_name": str(class_name),
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "sensitivity": sensitivity,
                "recall": sensitivity,
                "specificity": specificity,
                "f1": f1,
            }
        )

    return {
        "confusion_matrix": {
            "labels": list(class_names),
            "rows": "true_labels",
            "columns": "predicted_labels",
            "matrix": matrix,
        },
        "per_class": per_class,
        "macro_precision": _mean(item["precision"] for item in per_class),
        "macro_sensitivity": _mean(item["sensitivity"] for item in per_class),
        "macro_recall": _mean(item["recall"] for item in per_class),
        "macro_specificity": _mean(item["specificity"] for item in per_class),
        "macro_f1": _mean(item["f1"] for item in per_class),
        "weighted_precision": _weighted_mean(per_class, "precision"),
        "weighted_sensitivity": _weighted_mean(per_class, "sensitivity"),
        "weighted_recall": _weighted_mean(per_class, "recall"),
        "weighted_specificity": _weighted_mean(per_class, "specificity"),
        "weighted_f1": _weighted_mean(per_class, "f1"),
    }


def _progress(dataloader, split_name: str):
    from tqdm import tqdm

    return tqdm(dataloader, desc=split_name, leave=False)


def _set_progress_metrics(progress, total_loss: float, targets: Sequence[int], predictions: Sequence[int]) -> None:
    examples = len(targets)
    correct = sum(1 for true, pred in zip(targets, predictions) if true == pred)
    progress.set_postfix(loss=_safe_div(total_loss, examples), acc=_safe_div(correct, examples))


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return _safe_div(sum(values), len(values))


def _weighted_mean(per_class: Sequence[dict[str, object]], key: str) -> float:
    total_support = sum(int(item["support"]) for item in per_class)
    weighted_sum = sum(float(item[key]) * int(item["support"]) for item in per_class)
    return _safe_div(weighted_sum, total_support)


def _format_metrics_table(split_name: str, metrics, report: dict[str, object]) -> str:
    lines = [
        "| Split | Loss | Accuracy | Macro Precision | Macro Sensitivity | Macro Specificity | Macro F1 | Weighted F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {split_name} | {metrics.loss:.4f} | {metrics.accuracy:.4f} | "
            f"{report['macro_precision']:.4f} | {report['macro_sensitivity']:.4f} | "
            f"{report['macro_specificity']:.4f} | {report['macro_f1']:.4f} | "
            f"{report['weighted_f1']:.4f} |"
        ),
        "",
        "| Class | Support | Precision | Sensitivity | Specificity | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["per_class"]:
        lines.append(
            f"| {item['class_name']} | {item['support']} | {item['precision']:.4f} | "
            f"{item['sensitivity']:.4f} | {item['specificity']:.4f} | {item['f1']:.4f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
