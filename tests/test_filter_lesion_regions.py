import json
import sys
import types

import pytest
import torch

from xdl_densecaps.filter_lesion_regions import (
    choose_k_by_calinski_harabasz,
    output_filename,
    select_highest_max_similarity_candidate,
    select_lowest_similarity_candidate,
    write_metadata,
)


def test_select_lowest_similarity_candidate_uses_smallest_centroid_sum():
    normal_centroids = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    features = [
        torch.tensor([0.9, 0.8]),
        torch.tensor([0.2, 0.1]),
        torch.tensor([0.6, 0.5]),
    ]

    selected = select_lowest_similarity_candidate(features, normal_centroids)

    assert selected is not None
    assert selected.candidate_index == 1
    assert selected.score == pytest.approx(0.3)


def test_select_highest_max_similarity_candidate_uses_closest_single_centroid():
    normal_centroids = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    features = [
        torch.tensor([0.8, 0.8]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.6, 0.6]),
    ]

    selected = select_highest_max_similarity_candidate(features, normal_centroids)

    assert selected is not None
    assert selected.candidate_index == 1
    assert selected.score == pytest.approx(1.0)


def test_metadata_record_can_point_to_saved_output(tmp_path):
    output_path = tmp_path / "images" / output_filename(tmp_path / "image 1.jpg", 4, 2)
    normal_output_path = tmp_path / "normal_images" / output_filename(tmp_path / "normal 1.jpg", 5, 1)
    output_path.parent.mkdir(parents=True)
    normal_output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"fake-image")
    normal_output_path.write_bytes(b"fake-normal-image")

    metadata_path = tmp_path / "metadata.json"
    write_metadata(
        metadata_path,
        {
            "settings": {"output_image": "zoomed_masked_crop"},
            "normal_records": [{"output_path": str(normal_output_path)}],
            "records": [{"output_path": str(output_path)}],
        },
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["records"][0]["output_path"] == str(output_path)
    assert metadata["normal_records"][0]["output_path"] == str(normal_output_path)
    assert output_path.exists()
    assert normal_output_path.exists()


def test_kmeans_selection_tolerates_broken_psutil(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace())

    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )

    selected_k, ch_scores = choose_k_by_calinski_harabasz(
        features,
        max_k=3,
        random_state=42,
    )

    assert selected_k in {2, 3}
    assert ch_scores
