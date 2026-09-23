import json
import os

import numpy as np
import pytest
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

from keras_climate.datasets.copernicus.flood_s1 import CopernicusBenchFloodS1
from keras_climate.datasets.errors import DatasetNotFoundError

GRID_ID = "e2eb30d2c5eb52669c5e6a9db8973835"
GRID_PATH = "1111013/01/e2eb30d2c5eb52669c5e6a9db8973835"


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    directory = os.path.join(root, "flood_s1")
    subdir = os.path.join(directory, "data", GRID_PATH)
    os.makedirs(subdir, exist_ok=True)

    size = 8
    profile = {
        "driver": "GTiff",
        "width": size,
        "height": size,
        "count": 1,
        "crs": CRS.from_epsg(3857),
        "transform": Affine(10.0, 0.0, -10003405.0, 0.0, -10.0, 4730925.0),
    }

    # SL1/MS1 VV/VH bands, dated files
    names = [
        "MS1_IVV_1111013_01_20170504",
        "MS1_IVH_1111013_01_20170504",
        "SL1_IVV_1111013_01_20170422",
        "SL1_IVH_1111013_01_20170422",
    ]
    for name in names:
        profile["dtype"] = "float32"
        data = np.random.random(size=(size, size)).astype("float32")
        with rasterio.open(os.path.join(subdir, f"{name}.tif"), "w", **profile) as dst:
            dst.write(data, 1)

    # Mask
    profile["dtype"] = "uint8"
    mask = np.random.randint(0, 3, size=(size, size), dtype="uint8")
    with rasterio.open(
        os.path.join(subdir, "MK0_MLU_1111013_01_20170504.tif"), "w", **profile
    ) as dst:
        dst.write(mask, 1)

    metadata = {GRID_ID: {"path": GRID_PATH}}
    for split in ("train", "val", "test"):
        with open(os.path.join(directory, f"grid_dict_{split}.json"), "w") as f:
            json.dump(metadata, f)

    return root


def test_getitem(prepared_root):
    ds = CopernicusBenchFloodS1(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, 8, 8, 2)
    assert tuple(sample["mask"].shape) == (8, 8)
    assert "lat" in sample and "lon" in sample and "time" in sample


def test_len(prepared_root):
    ds = CopernicusBenchFloodS1(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_band_subset(prepared_root):
    ds = CopernicusBenchFloodS1(
        root=prepared_root, split="train", bands=("VV",), download=False
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, 8, 8, 1)


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchFloodS1(root=str(tmp_path), download=False)
