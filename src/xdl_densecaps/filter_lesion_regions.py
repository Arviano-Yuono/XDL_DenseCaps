"""Create a filtered lesion-region dataset from a trained classifier."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from xdl_densecaps.config import DEFAULT_CONFIG_PATH, load_config
from xdl_densecaps.datasets import CLASS_NAMES, BinaryNormalLesionDataset
from xdl_densecaps.lesion.components import SquareConnectedComponents
from xdl_densecaps.lesion.extraction import LesionCandidate, ZoomedMaskCropExtractor
from xdl_densecaps.training import (
    build_classifier,
    checkpoint_path,
    load_checkpoint,
    resolve_data_root,
    select_device,
)


NORMAL_LABEL = 0
LESION_LABEL = 1


@dataclass(frozen=True)
class CandidateScore:
    """Similarity score for one candidate region."""

    candidate_index: int
    score: float
    centroid_similarities: list[float]


class IndexedDataset(Dataset[tuple[Tensor, int, int]]):
    """Wrap a dataset so each batch still knows the original sample index."""

    def __init__(self, dataset: BinaryNormalLesionDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        image, label = self.dataset[index]
        return image, label, index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter lesion regions across the whole configured dataset.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to save filtered lesion crops. Defaults to <runtime.output_dir>/filtered_lesion_regions.",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="Grad-CAM threshold.")
    parser.add_argument("--min-area", type=int, default=3, help="Minimum low-res component area.")
    parser.add_argument("--max-k", type=int, default=8, help="Maximum KMeans clusters for normal regions.")
    parser.add_argument("--grad-cam-layer", default="denseblock4", help="DenseNet backbone layer for Grad-CAM.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    device = select_device(config.runtime.device)
    data_root = resolve_data_root(config.data.root_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.runtime.output_dir) / "filtered_lesion_regions"

    transform = transforms.Compose(
        [
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = BinaryNormalLesionDataset(data_root, transform=transform)
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
    )

    model = build_classifier(
        config,
        use_pretrained=False,
        use_backbone_checkpoint=False,
    ).to(device)
    best_checkpoint_path = checkpoint_path(config)
    load_checkpoint(best_checkpoint_path, model, device)
    model.eval()

    component_finder = SquareConnectedComponents(min_area=args.min_area)
    crop_extractor = ZoomedMaskCropExtractor()

    normal_features = collect_normal_features(
        model=model,
        loader=loader,
        device=device,
        threshold=args.threshold,
        grad_cam_layer=args.grad_cam_layer,
        component_finder=component_finder,
        crop_extractor=crop_extractor,
    )
    normal_centroids, selected_k, ch_scores = fit_normal_centroids(
        normal_features,
        max_k=args.max_k,
        random_state=config.data.seed,
    )

    records = save_filtered_lesion_regions(
        model=model,
        loader=loader,
        dataset=dataset,
        device=device,
        output_dir=output_dir,
        threshold=args.threshold,
        grad_cam_layer=args.grad_cam_layer,
        component_finder=component_finder,
        crop_extractor=crop_extractor,
        normal_centroids=normal_centroids,
    )

    metadata = {
        "settings": {
            "config_path": str(Path(args.config)),
            "data_root": str(data_root),
            "checkpoint_path": str(best_checkpoint_path),
            "class_names": list(CLASS_NAMES),
            "image_size": config.data.image_size,
            "threshold": args.threshold,
            "min_area": args.min_area,
            "max_k": args.max_k,
            "selected_k": selected_k,
            "grad_cam_layer": args.grad_cam_layer,
            "normal_reference": "correctly_predicted_normal_candidates",
            "output_image": "zoomed_masked_crop",
        },
        "normal_reference": {
            "candidate_count": len(normal_features),
            "ch_scores": {str(k): score for k, score in ch_scores.items()},
        },
        "records": records,
    }
    write_metadata(output_dir / "metadata.json", metadata)

    print(f"Normal reference candidates: {len(normal_features)}")
    print(f"Selected K: {selected_k}")
    print(f"Saved filtered lesion crops: {len(records)}")
    print(f"Output directory: {output_dir}")
    return 0


def collect_normal_features(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    grad_cam_layer: str,
    component_finder: SquareConnectedComponents,
    crop_extractor: ZoomedMaskCropExtractor,
) -> list[Tensor]:
    normal_features: list[Tensor] = []

    for images, labels, _indices in tqdm(loader, desc="Normal reference", leave=False):
        cams, scores = compute_grad_cam(
            model,
            images.to(device),
            target_class=LESION_LABEL,
            layer_name=grad_cam_layer,
        )
        predictions = scores.argmax(dim=1).cpu()

        for batch_index in range(images.size(0)):
            true_label = int(labels[batch_index])
            pred_label = int(predictions[batch_index])
            if true_label != NORMAL_LABEL or pred_label != true_label:
                continue

            candidates = extract_candidates(
                image=images[batch_index],
                attention_map=cams[batch_index],
                threshold=threshold,
                component_finder=component_finder,
                crop_extractor=crop_extractor,
            )
            for candidate in candidates:
                normal_features.append(candidate_feature(model, candidate.image, device))

    return normal_features


def save_filtered_lesion_regions(
    *,
    model: nn.Module,
    loader: DataLoader,
    dataset: BinaryNormalLesionDataset,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    grad_cam_layer: str,
    component_finder: SquareConnectedComponents,
    crop_extractor: ZoomedMaskCropExtractor,
    normal_centroids: Tensor,
) -> list[dict[str, object]]:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for images, labels, indices in tqdm(loader, desc="Lesion filtering", leave=False):
        cams, scores = compute_grad_cam(
            model,
            images.to(device),
            target_class=LESION_LABEL,
            layer_name=grad_cam_layer,
        )
        predictions = scores.argmax(dim=1).cpu()

        for batch_index in range(images.size(0)):
            true_label = int(labels[batch_index])
            pred_label = int(predictions[batch_index])
            if true_label != LESION_LABEL or pred_label != true_label:
                continue

            candidates = extract_candidates(
                image=images[batch_index],
                attention_map=cams[batch_index],
                threshold=threshold,
                component_finder=component_finder,
                crop_extractor=crop_extractor,
            )
            if not candidates:
                continue

            features = [candidate_feature(model, candidate.image, device) for candidate in candidates]
            best_score = select_lowest_similarity_candidate(features, normal_centroids)
            if best_score is None:
                continue

            dataset_index = int(indices[batch_index])
            sample = dataset.samples[dataset_index]
            selected_candidate = candidates[best_score.candidate_index]
            output_path = image_dir / output_filename(sample.path, dataset_index, best_score.candidate_index)
            save_tensor_image(selected_candidate.image, output_path)

            records.append(
                {
                    "dataset_index": dataset_index,
                    "original_path": str(sample.path),
                    "output_path": str(output_path),
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "pred_label": pred_label,
                    "pred_class": CLASS_NAMES[pred_label],
                    "pred_score": float(scores[batch_index, pred_label].detach().cpu()),
                    "candidate_index": best_score.candidate_index,
                    "normal_similarity_score": best_score.score,
                    "centroid_similarities": best_score.centroid_similarities,
                    "bbox_xyxy_resized": list(selected_candidate.bbox_xyxy),
                }
            )

    return records


def compute_grad_cam(
    model: nn.Module,
    images: Tensor,
    *,
    target_class: int,
    layer_name: str,
) -> tuple[Tensor, Tensor]:
    """Return normalized low-res Grad-CAM maps and model scores for a batch."""

    target_layer = get_grad_cam_layer(model, layer_name)
    activations: dict[str, Tensor] = {}
    gradients: dict[str, Tensor] = {}

    def save_activation(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        scores = model(images)
        scores[:, target_class].sum().backward()
    finally:
        handle.remove()

    if "value" not in activations or "value" not in gradients:
        raise RuntimeError(f"Grad-CAM hook did not capture activations for layer {layer_name!r}.")

    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations["value"]).sum(dim=1))
    return normalize_attention_map(cam).detach().cpu(), scores.detach().cpu()


def get_grad_cam_layer(model: nn.Module, layer_name: str) -> nn.Module:
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "features"):
        raise ValueError("Lesion filtering expects a model with a DenseNet feature backbone.")

    if layer_name == "backbone":
        return model.backbone

    layers = dict(model.backbone.features.named_children())
    if layer_name not in layers:
        options = ["backbone", *layers.keys()]
        raise ValueError(f"Unknown Grad-CAM layer {layer_name!r}. Options: {options}")
    return layers[layer_name]


def normalize_attention_map(attention_map: Tensor, eps: float = 1e-8) -> Tensor:
    if attention_map.dim() != 3:
        raise ValueError(f"Expected attention map [B, H, W], got {tuple(attention_map.shape)}")

    mins = attention_map.amin(dim=(1, 2), keepdim=True)
    maxs = attention_map.amax(dim=(1, 2), keepdim=True)
    return (attention_map - mins) / (maxs - mins + eps)


def extract_candidates(
    *,
    image: Tensor,
    attention_map: Tensor,
    threshold: float,
    component_finder: SquareConnectedComponents,
    crop_extractor: ZoomedMaskCropExtractor,
) -> list[LesionCandidate]:
    binary_mask = attention_map > threshold
    low_res_masks = component_finder(binary_mask)
    return [crop_extractor(image, low_res_mask) for low_res_mask in low_res_masks]


def candidate_feature(model: nn.Module, candidate_image: Tensor, device: torch.device) -> Tensor:
    with torch.no_grad():
        features = model.backbone(candidate_image.unsqueeze(0).to(device))
    features = features.reshape(features.size(0), -1)
    return F.normalize(features, p=2, dim=1).squeeze(0).detach().cpu()


def fit_normal_centroids(
    normal_features: list[Tensor],
    *,
    max_k: int,
    random_state: int,
) -> tuple[Tensor, int, dict[int, float]]:
    if not normal_features:
        raise ValueError(
            "No correctly predicted normal candidate regions were found. "
            "Try a lower threshold/min-area, check model predictions, or use a larger dataset."
        )

    feature_matrix = torch.stack([feature.float() for feature in normal_features], dim=0)
    selected_k, ch_scores = choose_k_by_calinski_harabasz(
        feature_matrix,
        max_k=max_k,
        random_state=random_state,
    )

    kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init="auto")
    kmeans.fit(feature_matrix.numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    return F.normalize(centroids, p=2, dim=1), selected_k, ch_scores


def choose_k_by_calinski_harabasz(
    feature_matrix: Tensor,
    *,
    max_k: int,
    random_state: int,
) -> tuple[int, dict[int, float]]:
    num_samples = feature_matrix.size(0)
    if num_samples < 3:
        return 1, {}

    max_valid_k = min(max_k, num_samples - 1)
    if max_valid_k < 2:
        return 1, {}

    features_np = feature_matrix.numpy()
    ch_scores: dict[int, float] = {}
    for k in range(2, max_valid_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(features_np)
        if len(set(labels.tolist())) < 2:
            continue
        ch_scores[k] = float(calinski_harabasz_score(features_np, labels))

    if not ch_scores:
        return 1, {}
    return max(ch_scores, key=ch_scores.get), ch_scores


def score_candidate_features(features: list[Tensor], normal_centroids: Tensor) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for candidate_index, feature in enumerate(features):
        similarities = torch.matmul(normal_centroids, feature.float())
        scores.append(
            CandidateScore(
                candidate_index=candidate_index,
                score=float(similarities.sum().item()),
                centroid_similarities=[float(value) for value in similarities.tolist()],
            )
        )
    return scores


def select_lowest_similarity_candidate(
    features: list[Tensor],
    normal_centroids: Tensor,
) -> CandidateScore | None:
    scores = score_candidate_features(features, normal_centroids)
    if not scores:
        return None
    return min(scores, key=lambda score: score.score)


def output_filename(original_path: Path, dataset_index: int, candidate_index: int) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", original_path.stem).strip("._")
    if not safe_stem:
        safe_stem = "image"
    return f"{dataset_index:06d}_{safe_stem}_candidate{candidate_index:02d}.png"


def save_tensor_image(image: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(image.detach().cpu().clamp(0.0, 1.0)).save(path)


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
