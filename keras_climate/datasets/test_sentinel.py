import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.errors import RGBBandsMissingError
from keras_climate.datasets.sentinel import Sentinel1, Sentinel2


def _write_tif(path, size=8, value=10):
    transform = from_origin(0, size, 1, 1)
    data = np.full((1, size, size), value, dtype="uint16")
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
        dst.write(data)


@pytest.fixture
def sentinel1_root(tmp_path):
    root = str(tmp_path)
    fname = "S1A_IW_20210101T000000_DVP_RTC30_G_gpuned_0000_VV.tif"
    _write_tif(os.path.join(root, fname))
    fname_vh = fname.replace("_VV.tif", "_VH.tif")
    _write_tif(os.path.join(root, fname_vh))
    return root


@pytest.fixture
def sentinel2_root(tmp_path):
    root = str(tmp_path)
    subdir = os.path.join(root, "R10m")
    os.makedirs(subdir, exist_ok=True)
    for band in ("B04", "B03", "B02"):
        fname = f"T32UPB_20210101T000000_{band}_10m.tif"
        _write_tif(os.path.join(subdir, fname))
    return root


def test_sentinel1_getitem(sentinel1_root):
    ds = Sentinel1(paths=sentinel1_root, bands=["VV", "VH"], res=1)
    x, y, t = ds.bounds
    sample = ds[x, y, t]
    assert tuple(sample["image"].shape) == (8, 8, 2)


def test_sentinel1_mixed_polarization_raises():
    with pytest.raises(AssertionError):
        Sentinel1(bands=["VV", "HH"])


def test_sentinel1_empty_bands_raises():
    with pytest.raises(AssertionError):
        Sentinel1(bands=[])


def test_sentinel1_plot_single_band(sentinel1_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Sentinel1(paths=sentinel1_root, bands=["VV"])
    x, y, t = ds.bounds
    ds.plot(ds[x, y, t])
    plt.close()


def test_sentinel1_plot_dual_band(sentinel1_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Sentinel1(paths=sentinel1_root, bands=["VV", "VH"])
    x, y, t = ds.bounds
    ds.plot(ds[x, y, t])
    plt.close()


def test_sentinel2_getitem(sentinel2_root):
    ds = Sentinel2(paths=sentinel2_root, bands=("B04", "B03", "B02"), res=1)
    x, y, t = ds.bounds
    sample = ds[x, y, t]
    assert tuple(sample["image"].shape) == (8, 8, 3)


def test_sentinel2_plot(sentinel2_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Sentinel2(paths=sentinel2_root, bands=("B04", "B03", "B02"))
    x, y, t = ds.bounds
    ds.plot(ds[x, y, t])
    plt.close()


def test_sentinel2_plot_missing_rgb_raises(sentinel2_root):
    ds = Sentinel2(paths=sentinel2_root, bands=("B04",))
    x, y, t = ds.bounds
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[x, y, t])
