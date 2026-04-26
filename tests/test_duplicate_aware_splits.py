from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from xdl_densecaps.datasets.binary_image_dataset import ImageSample
from xdl_densecaps.training import create_split_indices


class DummyDataset:
    def __init__(self, samples):
        self.samples = samples


def test_split_indices_keep_duplicate_files_together(tmp_path):
    samples = [
        ImageSample(_write_file(tmp_path, "normal_a.jpg", b"normal-a"), 0),
        ImageSample(_write_file(tmp_path, "normal_a_copy.jpg", b"normal-a"), 0),
        ImageSample(_write_file(tmp_path, "normal_b.jpg", b"normal-b"), 0),
        ImageSample(_write_file(tmp_path, "normal_c.jpg", b"normal-c"), 0),
        ImageSample(_write_file(tmp_path, "lesion_a.jpg", b"lesion-a"), 1),
        ImageSample(_write_file(tmp_path, "lesion_a_copy.jpg", b"lesion-a"), 1),
        ImageSample(_write_file(tmp_path, "lesion_b.jpg", b"lesion-b"), 1),
        ImageSample(_write_file(tmp_path, "lesion_c.jpg", b"lesion-c"), 1),
    ]
    dataset = DummyDataset(samples)

    splits = create_split_indices(dataset, val_ratio=0.25, test_ratio=0.25, seed=7)

    split_by_index = {
        index: split_name
        for split_name, indices in {
            "train": splits.train,
            "val": splits.val,
            "test": splits.test,
        }.items()
        for index in indices
    }
    split_names_by_hash = defaultdict(set)

    for index, sample in enumerate(samples):
        digest = sha256(sample.path.read_bytes()).hexdigest()
        split_names_by_hash[digest].add(split_by_index[index])

    assert set(split_by_index) == set(range(len(samples)))
    assert all(len(split_names) == 1 for split_names in split_names_by_hash.values())


def _write_file(tmp_path: Path, name: str, contents: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(contents)
    return path
