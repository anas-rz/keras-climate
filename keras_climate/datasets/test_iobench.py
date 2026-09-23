import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

# IOBench depends on the CDL and Landsat9 datasets, which are ported by other
# files in this same parallel porting effort and may not exist yet in this
# environment. Skip this whole module until both land.
pytest.importorskip("keras_climate.datasets.cdl")
pytest.importorskip("keras_climate.datasets.landsat")

from keras_climate.datasets import DatasetNotFoundError  # noqa: E402
from keras_climate.datasets.iobench import IOBench  # noqa: E402
from keras_climate.datasets.landsat import Landsat9  # noqa: E402


def _write_band_tif(path, value):
    transform = from_origin(0, 10, 1, 1)
    data = np.full((1, 10, 10), value, dtype="uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="uint16",
        crs="EPSG:32612",
        transform=transform,
    ) as dst:
        dst.write(data)


def _write_cdl_tif(path, value):
    transform = from_origin(0, 10, 1, 1)
    data = np.full((1, 10, 10), value, dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="uint8",
        crs="EPSG:32612",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    split_dir = os.path.join(root, "preprocessed")
    os.makedirs(split_dir, exist_ok=True)

    bands = [*Landsat9.default_bands, "SR_QA_AEROSOL"]
    for band in bands:
        fname = f"LC09_L2SP_012025_20230101_20230102_02_T1_{band}.TIF"
        _write_band_tif(os.path.join(split_dir, fname), 100)

    # 0 is always a valid (background) CDL class
    _write_cdl_tif(os.path.join(split_dir, "2023_30m_cdls.tif"), 0)

    return root


def test_getitem(prepared_root):
    ds = IOBench(root=prepared_root, split="preprocessed", download=False)
    sample = ds[ds.bounds]
    assert "image" in sample
    assert "mask" in sample


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        IOBench(root=prepared_root, split="bogus", download=False)


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        IOBench(root=str(tmp_path), split="preprocessed", download=False)
