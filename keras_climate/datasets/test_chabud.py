import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.chabud import ChaBuD

SIZE = 8
NUM_CHANNELS = 12


@pytest.fixture
def prepared_root(tmp_path):
    data = np.random.randint(4096, size=(SIZE, SIZE, NUM_CHANNELS)).astype("uint16")
    gt = np.random.randint(2, size=(SIZE, SIZE, 1)).astype("uint16")

    path = os.path.join(str(tmp_path), "train_eval.hdf5")
    uris = ["uuid_a", "uuid_b", "uuid_c"]
    folds = [1, 0, 2]

    with h5py.File(path, "a") as f:
        for uri, fold in zip(uris, folds):
            sample = f.create_group(uri)
            sample.attrs.create("fold", data=np.int64(fold))
            sample.create_dataset("pre_fire", data=data)
            sample.create_dataset("post_fire", data=data)
            sample.create_dataset("mask", data=gt)

    return str(tmp_path)


def test_getitem(prepared_root):
    ds = ChaBuD(root=prepared_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, SIZE, SIZE, NUM_CHANNELS)
    assert tuple(sample["mask"].shape) == (1, SIZE, SIZE)


def test_len(prepared_root):
    ds_train = ChaBuD(root=prepared_root, split="train")
    ds_val = ChaBuD(root=prepared_root, split="val")
    assert len(ds_train) == 2
    assert len(ds_val) == 1


def test_band_subset(prepared_root):
    ds = ChaBuD(root=prepared_root, split="train", bands=("B04", "B03", "B02"))
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, SIZE, SIZE, 3)


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        ChaBuD(root=prepared_root, split="bogus")


def test_invalid_bands(prepared_root):
    with pytest.raises(AssertionError):
        ChaBuD(root=prepared_root, bands=("BOGUS",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        ChaBuD(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ChaBuD(root=prepared_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()


def test_plot_missing_rgb(prepared_root):
    ds = ChaBuD(root=prepared_root, split="train", bands=("B01",))
    with pytest.raises(ValueError):
        ds.plot(ds[0])
