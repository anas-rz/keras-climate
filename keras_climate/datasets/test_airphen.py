import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import RGBBandsMissingError
from keras_climate.datasets._test_helpers import assert_dtype
from keras_climate.datasets.airphen import Airphen


@pytest.fixture
def prepared_root(tmp_path):
    transform = from_origin(0, 8, 1, 1)
    data = np.random.randint(0, 4096, (8, 8, 8)).astype("uint16")
    path = os.path.join(str(tmp_path), "airphen.tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=8,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = Airphen(prepared_root)
    x = ds[ds.bounds]
    assert tuple(x["image"].shape) == (8, 8, 8)
    assert_dtype(x["image"], "float32")


def test_len(prepared_root):
    ds = Airphen(prepared_root)
    assert len(ds) == 1


def test_band_subset(prepared_root):
    ds = Airphen(prepared_root, bands=("B1", "B2", "B3"))
    x = ds[ds.bounds]
    assert tuple(x["image"].shape) == (8, 8, 3)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Airphen(prepared_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()


def test_plot_missing_rgb(prepared_root):
    ds = Airphen(prepared_root, bands=("B2",))
    x = ds[ds.bounds]
    with pytest.raises(RGBBandsMissingError):
        ds.plot(x)
