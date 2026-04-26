import json

import pytest
import torch

from xdl_densecaps.filter_lesion_regions import (
    output_filename,
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


def test_metadata_record_can_point_to_saved_output(tmp_path):
    output_path = tmp_path / "images" / output_filename(tmp_path / "image 1.jpg", 4, 2)
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"fake-image")

    metadata_path = tmp_path / "metadata.json"
    write_metadata(
        metadata_path,
        {
            "settings": {"output_image": "zoomed_masked_crop"},
            "records": [{"output_path": str(output_path)}],
        },
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["records"][0]["output_path"] == str(output_path)
    assert output_path.exists()
