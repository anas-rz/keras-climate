import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.errors import DatasetNotFoundError
from keras_climate.datasets.copernicus.aq_no2_s5p import CopernicusBenchAQNO2S5P

TRANSFORM = from_origin(10, 50, 0.01, 0.01)


def _write_tif(path, bands=1, size=8, dtype="float32"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.random.rand(bands, size, size).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)


def _prepare_root(tmp_path, split="train"):
    root = str(tmp_path)
    directory = os.path.join(root, "airquality_s5p", "no2")
    pid = "loc_0001"

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{split}.csv"), "w") as f:
        f.write(pid + "\n")

    annual_path = os.path.join(
        directory, "s5p_annual", pid, "2021-01-01_2021-12-31.tif"
    )
    _write_tif(annual_path)

    for file in (
        "2021-01-01_2021-04-01.tif",
        "2021-04-01_2021-07-01.tif",
        "2021-07-01_2021-10-01.tif",
        "2021-10-01_2021-12-31.tif",
    ):
        _write_tif(os.path.join(directory, "s5p_seasonal", pid, file))

    _write_tif(os.path.join(directory, "label_annual", f"{pid}.tif"))

    return root


def test_getitem_annual(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchAQNO2S5P(root=root, split="train", mode="annual")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 1)
    assert "mask" in sample
    assert "lat" in sample and "lon" in sample
    assert "time" in sample


def test_getitem_seasonal(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchAQNO2S5P(root=root, split="train", mode="seasonal")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 8, 8, 1)
    assert "mask" in sample


def test_len(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchAQNO2S5P(root=root, split="train")
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchAQNO2S5P(root=str(tmp_path), download=False)
