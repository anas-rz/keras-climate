import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.cabuar import CaBuAr

SIZE = 8
NUM_CHANNELS = 12


@pytest.fixture
def prepared_root(tmp_path):
    data = np.random.randint(4096, size=(SIZE, SIZE, NUM_CHANNELS)).astype("uint16")
    gt = np.random.randint(2, size=(SIZE, SIZE, 1)).astype("uint16")

    filenames = ["512x512.hdf5", "chabud_test.h5"]
    uris = ["uuid_a", "uuid_b", "uuid_c", "uuid_d"]
    folds = [1, 0] + ["chabud", "chabud"]
    files = ["512x512.hdf5"] * 2 + ["chabud_test.h5"] * 2

    for filename in filenames:
        path = os.path.join(str(tmp_path), filename)
        with h5py.File(path, "a"):
            pass

    for uri, fold, filename in zip(uris, folds, files):
        path = os.path.join(str(tmp_path), filename)
        with h5py.File(path, "a") as f:
            sample = f.create_group(uri)
            sample.attrs.create("fold", data=fold if fold == "chabud" else np.int64(fold))
            sample.create_dataset("pre_fire", data=data)
            sample.create_dataset("post_fire", data=data)
            sample.create_dataset("mask", data=gt)

    return str(tmp_path)


def test_getitem(prepared_root):
    ds = CaBuAr(root=prepared_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, SIZE, SIZE, NUM_CHANNELS)
    assert tuple(sample["mask"].shape) == (1, SIZE, SIZE)


def test_len(prepared_root):
    ds_train = CaBuAr(root=prepared_root, split="train")
    ds_val = CaBuAr(root=prepared_root, split="val")
    ds_test = CaBuAr(root=prepared_root, split="test")
    assert len(ds_train) == 1
    assert len(ds_val) == 1
    assert len(ds_test) == 2


def test_band_subset(prepared_root):
    ds = CaBuAr(root=prepared_root, split="train", bands=("B04", "B03", "B02"))
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, SIZE, SIZE, 3)


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        CaBuAr(root=prepared_root, split="bogus")


def test_invalid_bands(prepared_root):
    with pytest.raises(AssertionError):
        CaBuAr(root=prepared_root, bands=("BOGUS",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CaBuAr(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = CaBuAr(root=prepared_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()


def test_plot_missing_rgb(prepared_root):
    ds = CaBuAr(root=prepared_root, split="train", bands=("B01",))
    with pytest.raises(ValueError):
        ds.plot(ds[0])
