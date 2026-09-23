import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.quakeset import QuakeSet


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    filepath = os.path.join(root, QuakeSet.filename)

    data = np.random.randn(4, 4, 2).astype("float32")
    splits = {
        "train": ["611645479", "611658170"],
        "validation": ["611684805"],
        "test": ["611798698"],
    }

    with h5py.File(filepath, "w") as f:
        for split, keys in splits.items():
            for key in keys:
                sample = f.create_group(key)
                sample.attrs.create(name="magnitude", data=np.float32(1.2))
                sample.attrs.create(name="split", data=split)
                for i in range(2):
                    patch = sample.create_group(f"patch_{i}")
                    patch.create_dataset("before", data=data)
                    patch.create_dataset("pre", data=data)
                    patch.create_dataset("post", data=data)

    return root


def test_getitem(prepared_root):
    ds = QuakeSet(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 4, 4)
    assert "label" in sample
    assert "magnitude" in sample


def test_len(prepared_root):
    # 2 keys x 2 patches x 2 samples (positive + hard negative) = 8
    ds = QuakeSet(root=prepared_root, split="train", download=False)
    assert len(ds) == 8


def test_val_split(prepared_root):
    ds = QuakeSet(root=prepared_root, split="val", download=False)
    assert len(ds) == 4


def test_invalid_split():
    with pytest.raises(AssertionError):
        QuakeSet(split="foo")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        QuakeSet(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = QuakeSet(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
