import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.errors import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.copernicus.cloud_s2 import CopernicusBenchCloudS2

TRANSFORM = from_origin(10, 50, 0.01, 0.01)
N_BANDS = len(CopernicusBenchCloudS2.all_bands)


def _write_image(path, bands, size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.random.randint(0, 4000, (bands, size, size)).astype("uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype="uint16",
        crs="EPSG:4326",
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)


def _write_mask(path, size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.random.randint(0, 4, (1, size, size)).astype("uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)


def _prepare_root(tmp_path, split="train"):
    root = str(tmp_path)
    directory = os.path.join(root, "cloud_s2")
    os.makedirs(directory, exist_ok=True)

    pid = "ROI_00001__20180425T054022"
    _write_image(os.path.join(directory, "s2_toa", f"{pid}.tif"), bands=N_BANDS)
    _write_mask(os.path.join(directory, "cloud", f"{pid}.tif"))

    with open(os.path.join(directory, f"{split}.csv"), "w") as f:
        f.write(pid + "\n")

    return root


def test_getitem(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchCloudS2(root=root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, N_BANDS)
    assert tuple(sample["mask"].shape) == (8, 8)
    assert "lat" in sample and "lon" in sample
    assert "time" in sample


def test_len(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchCloudS2(root=root, split="train")
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchCloudS2(root=str(tmp_path), download=False)


def test_plot_rgb_missing_band_raises(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchCloudS2(root=root, split="train", bands=("B01",))
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = _prepare_root(tmp_path)
    ds = CopernicusBenchCloudS2(root=root, split="train")
    sample = ds[0]
    sample["prediction"] = sample["mask"]
    ds.plot(sample, suptitle="Test")
    plt.close()
