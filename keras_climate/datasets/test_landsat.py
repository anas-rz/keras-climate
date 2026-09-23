import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.landsat import Landsat8
from keras_climate.datasets.geo import IntersectionDataset, UnionDataset


def _write_band(path, size=8):
    transform = from_origin(0, size, 1, 1)
    data = np.random.randint(0, 255, (size, size)).astype("uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def prepared_root(tmp_path):
    base = str(tmp_path)
    name = "LC08_L2SP_023032_20210622_20210629_02_T1"
    for band in ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]:
        _write_band(os.path.join(base, f"{name}_{band}.TIF"))
    return base


def test_getitem(prepared_root):
    ds = Landsat8(prepared_root)
    sample = ds[ds.bounds]
    assert "image" in sample
    assert tuple(sample["image"].shape[-1:]) == (7,)


def test_len(prepared_root):
    ds = Landsat8(prepared_root)
    assert len(ds) == 1


def test_and(prepared_root):
    ds = Landsat8(prepared_root)
    assert isinstance(ds & ds, IntersectionDataset)


def test_or(prepared_root):
    ds = Landsat8(prepared_root)
    assert isinstance(ds | ds, UnionDataset)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Landsat8(prepared_root)
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()


def test_plot_wrong_bands(prepared_root):
    ds = Landsat8(prepared_root, bands=("SR_B1",))
    sample = ds[ds.bounds]
    with pytest.raises(RGBBandsMissingError):
        ds.plot(sample)


def test_no_data(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        Landsat8(str(tmp_path))
