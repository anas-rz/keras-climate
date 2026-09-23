import os

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.rwanda_field_boundary import RwandaFieldBoundary


@pytest.fixture
def prepared_root(tmp_path, monkeypatch):
    # Use a single chip per split so the fixture stays small.
    monkeypatch.setattr(RwandaFieldBoundary, "splits", {"train": 1, "test": 1})

    root = str(tmp_path)
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": 8,
        "height": 8,
        "count": 1,
        "crs": CRS.from_epsg(3857),
        "transform": Affine(4.0, 0.0, 0.0, 0.0, -4.0, 0.0),
    }
    data = np.random.randint(0, 1000, (8, 8)).astype("uint16")

    for split in ("train", "test"):
        for date in RwandaFieldBoundary.dates:
            path = os.path.join(root, "source", split, date)
            os.makedirs(path, exist_ok=True)
            for band in RwandaFieldBoundary.all_bands:
                fpath = os.path.join(path, f"00_{band}.tif")
                with rasterio.open(fpath, "w", **profile) as dst:
                    dst.write(data, 1)

    labels_path = os.path.join(root, "labels", "train")
    os.makedirs(labels_path, exist_ok=True)
    with rasterio.open(os.path.join(labels_path, "00.tif"), "w", **profile) as dst:
        dst.write((data % 2).astype("uint16"), 1)

    return root


def test_getitem_train(prepared_root):
    ds = RwandaFieldBoundary(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (6, 8, 8, 4)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_getitem_test_split_has_no_mask(prepared_root):
    ds = RwandaFieldBoundary(root=prepared_root, split="test", download=False)
    sample = ds[0]
    assert "mask" not in sample


def test_len(prepared_root):
    ds = RwandaFieldBoundary(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_invalid_split():
    with pytest.raises(AssertionError):
        RwandaFieldBoundary(split="foo")


def test_invalid_bands():
    with pytest.raises(AssertionError):
        RwandaFieldBoundary(bands=("B99",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        RwandaFieldBoundary(root=str(tmp_path), download=False)


def test_plot_rgb_missing_band_raises(prepared_root):
    ds = RwandaFieldBoundary(
        root=prepared_root, split="train", bands=("B01",), download=False
    )
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = RwandaFieldBoundary(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
