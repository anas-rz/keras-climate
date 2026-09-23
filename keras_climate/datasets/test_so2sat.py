import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.so2sat import So2Sat


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    path = os.path.join(root, "training.h5")
    n = 3
    with h5py.File(path, "w") as f:
        f.create_dataset("sen1", data=np.random.rand(n, 8, 8, 8))
        f.create_dataset("sen2", data=np.random.rand(n, 8, 8, 10))
        onehot = np.zeros((n, 17), dtype="float32")
        onehot[np.arange(n), np.arange(n) % 17] = 1
        f.create_dataset("label", data=onehot)
    return root


def test_getitem_shape_and_dtype(prepared_root):
    ds = So2Sat(root=prepared_root, version="2", split="train", checksum=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 18)
    assert "label" in sample


def test_len(prepared_root):
    ds = So2Sat(root=prepared_root, version="2", split="train", checksum=False)
    assert len(ds) == 3


def test_band_subset(prepared_root):
    ds = So2Sat(
        root=prepared_root,
        version="2",
        split="train",
        bands=So2Sat.BAND_SETS["s2"],
        checksum=False,
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 10)


def test_invalid_band_raises(prepared_root):
    with pytest.raises(ValueError):
        So2Sat(root=prepared_root, bands=("bogus",), checksum=False)


def test_invalid_version_raises():
    with pytest.raises(AssertionError):
        So2Sat(version="bogus")


def test_not_found_raises(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        So2Sat(root=str(tmp_path), version="2", split="train", checksum=False)


def test_plot_missing_rgb_raises(prepared_root):
    ds = So2Sat(
        root=prepared_root,
        version="2",
        split="train",
        bands=So2Sat.BAND_SETS["s1"],
        checksum=False,
    )
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = So2Sat(root=prepared_root, version="2", split="train", checksum=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
