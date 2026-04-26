from PIL import Image
from torchvision import transforms

from xdl_densecaps.datasets.binary_image_dataset import BinaryNormalLesionDataset


def test_dataset_accepts_old_lession_folder_as_lesion_class(tmp_path):
    _write_image(tmp_path / "normal" / "normal.jpg")
    _write_image(tmp_path / "lession" / "old_spelling.jpg")

    dataset = BinaryNormalLesionDataset(
        tmp_path,
        transform=transforms.ToTensor(),
    )

    assert dataset.class_names == ("normal", "lesion")
    assert dataset.class_counts() == {"normal": 1, "lesion": 1}
    assert [sample.label for sample in dataset.samples] == [1, 0]


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(path)
