import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.zuericrop import ZueriCrop

NUM_SAMPLES = 2
NUM_CHANNELS = 9
SIZE = 8
NUM_CLASSES = 10


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    rng = np.random.default_rng(0)

    data = rng.integers(0, 4096, size=(NUM_SAMPLES, 1, SIZE, SIZE, NUM_CHANNELS)).astype("float64")
    gt = rng.integers(0, NUM_CLASSES, size=(NUM_SAMPLES, SIZE, SIZE, 1)).astype("int16")
    gt_instance = rng.integers(0, NUM_CLASSES, size=(NUM_SAMPLES, SIZE, SIZE, 1)).astype("int32")

    with h5py.File(os.path.join(root, "ZueriCrop.hdf5"), "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("gt", data=gt)
        f.create_dataset("gt_instance", data=gt_instance)

    with open(os.path.join(root, "labels.csv"), "w") as f:
        f.write("")

    return root


def test_getitem(prepared_root):
    ds = ZueriCrop(root=prepared_root, download=False)
    sample = ds[0]
    assert "image" in sample
    assert "mask" in sample
    assert "bbox_xyxy" in sample
    assert "label" in sample

    assert sample["image"].ndim == 4
    assert sample["mask"].ndim == 3
    assert tuple(sample["mask"].shape)[-2:] == tuple(sample["image"].shape)[-3:-1]
    assert sample["bbox_xyxy"].ndim == 2
    assert tuple(sample["bbox_xyxy"].shape)[1] == 4
    assert sample["label"].ndim == 1


def test_len(prepared_root):
    ds = ZueriCrop(root=prepared_root, download=False)
    assert len(ds) == NUM_SAMPLES


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        ZueriCrop(root=str(tmp_path), download=False)


def test_invalid_bands():
    with pytest.raises(ValueError):
        ZueriCrop(bands=("OK", "BK"))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from keras import ops

    ds = ZueriCrop(root=prepared_root, download=False)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()

    sample["prediction"] = sample["mask"]
    ds.plot(sample, suptitle="prediction")
    plt.close()


def test_plot_rgb_missing_band(prepared_root):
    ds = ZueriCrop(root=prepared_root, bands=("B02",), download=False)
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0], time_step=0, suptitle="Single Band")
