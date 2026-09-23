import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    UnionDataset,
)
from keras_climate.datasets.enmap import EnMAP


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    fname = (
        "ENMAP01-____L2A-DT0000001053_20220611T072305Z_002_V010400_"
        "20231221T134421Z-SPECTRAL_IMAGE.tif"
    )
    path = os.path.join(root, fname)
    transform = from_origin(0, 8, 1, 1)
    data = np.random.randint(0, 5000, (3, 8, 8)).astype("uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return root


def test_getitem(prepared_root):
    ds = EnMAP(prepared_root, bands=("B1", "B2", "B3"))
    sample = ds[ds.bounds]
    assert "image" in sample
    assert tuple(sample["image"].shape) == (8, 8, 3)


def test_len(prepared_root):
    ds = EnMAP(prepared_root, bands=("B1", "B2", "B3"))
    assert len(ds) == 1


def test_and_or(prepared_root):
    ds = EnMAP(prepared_root, bands=("B1", "B2", "B3"))
    assert isinstance(ds & ds, IntersectionDataset)
    assert isinstance(ds | ds, UnionDataset)


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EnMAP(str(tmp_path), bands=("B1", "B2", "B3"))


def test_plot_missing_rgb_bands_raises(prepared_root):
    ds = EnMAP(prepared_root, bands=("B1", "B2", "B3"))
    x = ds[ds.bounds]
    with pytest.raises(RGBBandsMissingError):
        ds.plot(x)


def test_default_bands_excludes_water_vapor_bands():
    assert "B1" in EnMAP.default_bands
    assert "B136" not in EnMAP.default_bands
    assert len(EnMAP.default_bands) < len(EnMAP.all_bands)
