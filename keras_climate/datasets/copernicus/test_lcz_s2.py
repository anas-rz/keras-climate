import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from keras_climate.datasets.copernicus.lcz_s2 import CopernicusBenchLCZS2
from keras_climate.datasets.errors import DatasetNotFoundError

SIZE = 8
NUM_CLASSES = 17
NUM_SAMPLES = 3


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    np.random.seed(0)

    for split in ("train", "val", "test"):
        filename = str(tmp_path / f"lcz_{split}.h5")
        label = np.eye(NUM_CLASSES, dtype="u1")[
            np.random.choice(NUM_CLASSES, NUM_SAMPLES)
        ]
        sen2 = np.random.random(size=(NUM_SAMPLES, SIZE, SIZE, 10)).astype("<f4")
        with h5py.File(filename, "w") as f:
            f.create_dataset("label", data=label)
            f.create_dataset("sen2", data=sen2)

    return root


def test_getitem(prepared_root):
    ds = CopernicusBenchLCZS2(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (SIZE, SIZE, 10)
    assert "label" in sample


def test_band_subset(prepared_root):
    ds = CopernicusBenchLCZS2(
        root=prepared_root, split="train", bands=("B02", "B03", "B04"), download=False
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (SIZE, SIZE, 3)


def test_len(prepared_root):
    ds = CopernicusBenchLCZS2(root=prepared_root, split="train", download=False)
    assert len(ds) == NUM_SAMPLES


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchLCZS2(root=str(tmp_path), download=False)
