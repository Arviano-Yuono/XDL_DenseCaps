import json

from PIL import Image
from torchvision import transforms

from xdl_densecaps.datasets import PairedImageDataset


def test_paired_dataset_loads_multiclass_metadata_records(tmp_path):
    whole_a = _write_image(tmp_path / "whole" / "alpha.jpg")
    detail_a = _write_image(tmp_path / "detail" / "alpha.png")
    whole_b = _write_image(tmp_path / "whole" / "beta.jpg")
    detail_b = _write_image(tmp_path / "detail" / "beta.png")
    metadata_path = tmp_path / "pairs.json"
    metadata_path.write_text(
        json.dumps(
            {
                "settings": {"class_names": ["alpha", "beta", "gamma"]},
                "records": [
                    {
                        "original_path": str(whole_a),
                        "output_path": str(detail_a),
                        "true_class": "alpha",
                    },
                    {
                        "original_path": str(whole_b),
                        "output_path": str(detail_b),
                        "true_class": "beta",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = PairedImageDataset(
        metadata_path=metadata_path,
        transform=transforms.ToTensor(),
    )

    whole_image, detail_image, label = dataset[1]

    assert dataset.class_names == ("alpha", "beta", "gamma")
    assert dataset.class_counts() == {"alpha": 1, "beta": 1, "gamma": 0}
    assert label == 1
    assert whole_image.shape == (3, 4, 4)
    assert detail_image.shape == (3, 4, 4)


def test_paired_dataset_matches_mirrored_class_roots_with_different_detail_extension(tmp_path):
    _write_image(tmp_path / "whole" / "class_a" / "image_1.jpg")
    _write_image(tmp_path / "detail" / "class_a" / "image_1.png")

    dataset = PairedImageDataset(
        tmp_path / "whole",
        detail_root_dir=tmp_path / "detail",
        transform=transforms.ToTensor(),
    )

    assert len(dataset) == 1
    assert dataset.class_names == ("class_a",)
    assert dataset[0][2] == 0


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path)
    return path
