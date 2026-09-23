import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.errors import DatasetNotFoundError
from keras_climate.datasets.copernicus.bigearthnet_s2 import CopernicusBenchBigEarthNetS2

TRANSFORM = from_origin(10, 50, 0.01, 0.01)
N_CLASSES = len(CopernicusBenchBigEarthNetS2.classes)
N_BANDS = len(CopernicusBenchBigEarthNetS2.all_bands)


def _write_tif(path, bands, size=8, dtype="float32"):
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
    directory = os.path.join(root, "bigearthnet_s1s2")
    os.makedirs(directory, exist_ok=True)

    s1_name = "S1A_IW_GRDH_ABCD_20180425T054022.tif"
    s2_name = "S2A_MSIL2A_20180425T054022.tif"

    _write_tif(os.path.join(directory, "BigEarthNet-S1-5%", s1_name), bands=2)
    _write_tif(os.path.join(directory, "BigEarthNet-S2-5%", s2_name), bands=N_BANDS)

    labels = np.zeros(N_CLASSES, dtype="int64")
    labels[1] = 1
    labels[4] = 1
    row = {"s1_filename": s1_name, "s2_filename": s2_name}
    row.update({f"class_{i}": labels[i] for i in range(N_CLASSES)})
    df = pd.DataFrame([row])
    df.to_csv(os.path.join(directory, f"multilabel-{split}.csv"), index=False)

    return root


def test_getitem(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchBigEarthNetS2(root=root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, N_BANDS)
    assert tuple(sample["label"].shape) == (N_CLASSES,)
    assert "lat" in sample and "lon" in sample
    assert "time" in sample


def test_len(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchBigEarthNetS2(root=root, split="train")
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchBigEarthNetS2(root=str(tmp_path), download=False)
