import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.errors import DatasetNotFoundError
from keras_climate.datasets.copernicus.cloud_s3 import CopernicusBenchCloudS3

TRANSFORM = from_origin(10, 50, 0.01, 0.01)
N_BANDS = len(CopernicusBenchCloudS3.all_bands)


def _write_image(path, bands, size=8, dtype="float32"):
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


def _write_mask(path, n_classes, size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.random.randint(0, n_classes, (1, size, size)).astype("uint8")
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
    directory = os.path.join(root, "cloud_s3")
    os.makedirs(directory, exist_ok=True)

    filename = "S3A_OL_1_EFR____20180425T054022.tif"
    _write_image(os.path.join(directory, "s3_olci", filename), bands=N_BANDS)
    _write_mask(os.path.join(directory, "cloud_multi", filename), n_classes=6)
    _write_mask(os.path.join(directory, "cloud_binary", filename), n_classes=3)

    with open(os.path.join(directory, f"{split}.csv"), "w") as f:
        f.write(filename + "\n")

    return root


def test_getitem_multi(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchCloudS3(root=root, split="train", mode="multi")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, N_BANDS)
    assert tuple(sample["mask"].shape) == (8, 8)
    assert "time" in sample


def test_getitem_binary(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchCloudS3(root=root, split="train", mode="binary")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, N_BANDS)
    assert ds.classes == ("Invalid", "Clear", "Cloud")


def test_len(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchCloudS3(root=root, split="train")
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchCloudS3(root=str(tmp_path), download=False)
