import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.errors import DatasetNotFoundError
from keras_climate.datasets.copernicus.dfc2020_s1 import CopernicusBenchDFC2020S1

TRANSFORM = from_origin(10, 50, 0.01, 0.01)
N_BANDS = len(CopernicusBenchDFC2020S1.all_bands)


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
    directory = os.path.join(root, "dfc2020_s1s2")
    os.makedirs(directory, exist_ok=True)

    dfc_file = "dfc_0001.tif"
    s1_file = "s1_0001.tif"
    _write_image(os.path.join(directory, "s1", s1_file), bands=N_BANDS)
    _write_mask(
        os.path.join(directory, "dfc", dfc_file),
        n_classes=len(CopernicusBenchDFC2020S1.classes),
    )

    with open(os.path.join(directory, f"dfc-{split}-new.csv"), "w") as f:
        f.write(dfc_file + "\n")

    return root


def test_getitem(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchDFC2020S1(root=root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, N_BANDS)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchDFC2020S1(root=root, split="train")
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchDFC2020S1(root=str(tmp_path), download=False)
