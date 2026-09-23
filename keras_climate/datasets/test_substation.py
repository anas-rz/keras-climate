import os

import numpy as np
import pytest

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.substation import Substation


@pytest.fixture
def prepared_root(tmp_path):
    image_dir = os.path.join(str(tmp_path), "image_stack")
    mask_dir = os.path.join(str(tmp_path), "mask")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(3):
        fname = f"sample_{i}.npz"
        image = np.random.randint(0, 255, (4, 13, 8, 8)).astype("float32")
        mask = np.random.randint(0, 4, (8, 8)).astype("uint8")
        np.savez(os.path.join(image_dir, fname), arr_0=image)
        np.savez(os.path.join(mask_dir, fname), arr_0=mask)

    return str(tmp_path)


def test_getitem(prepared_root):
    ds = Substation(root=prepared_root, mask_2d=True, num_of_timepoints=4)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 8, 8, 13)
    assert tuple(sample["mask"].shape) == (8, 8, 2)


def test_getitem_1d_mask(prepared_root):
    ds = Substation(root=prepared_root, mask_2d=False, num_of_timepoints=4)
    sample = ds[0]
    assert tuple(sample["mask"].shape) == (8, 8)


def test_timepoint_aggregation(prepared_root):
    ds = Substation(root=prepared_root, timepoint_aggregation="median")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 13)

    ds = Substation(root=prepared_root, timepoint_aggregation="first")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 13)

    ds = Substation(root=prepared_root, timepoint_aggregation="concat")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 52)


def test_band_subset(prepared_root):
    ds = Substation(root=prepared_root, bands=("B4", "B3", "B2"))
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 8, 8, 3)


def test_len(prepared_root):
    ds = Substation(root=prepared_root)
    assert len(ds) == 3


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        Substation(root=str(tmp_path), download=False)


def test_plot_rgb_missing_band_raises(prepared_root):
    ds = Substation(root=prepared_root, bands=("B1",))
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Substation(root=prepared_root, timepoint_aggregation="first")
    ds.plot(ds[0], suptitle="Test")
    plt.close()

    ds = Substation(root=prepared_root)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
