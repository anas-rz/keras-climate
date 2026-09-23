import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets._test_helpers import assert_dtype, assert_int64_dtype
from keras_climate.datasets.spacenet.spacenet4 import SpaceNet4

TRANSFORM = from_origin(0, 4, 1, 1)
CRS = "EPSG:4326"

GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1, 1], [1, 3], [3, 3], [3, 1], [1, 1]]],
            },
            "properties": {},
        }
    ],
}


def _write_tif(path, count, height=4, width=4, dtype="uint8"):
    data = np.random.randint(0, 255, (count, height, width)).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=CRS,
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)


def _write_geojson(path, empty=False):
    if empty:
        open(path, "w").close()
    else:
        with open(path, "w") as f:
            json.dump(GEOJSON, f)


def _build_root(tmp_path, split, num=2):
    # SpaceNet4's directory_glob is "**/{product}" with no {aoi} placeholder, so
    # images/masks live under an arbitrary nested catalog-id directory.
    cat_root = os.path.join(str(tmp_path), "SN4_buildings", split, "nadir7_catid_XYZ")
    image_dir = os.path.join(cat_root, "MS")
    os.makedirs(image_dir, exist_ok=True)
    for i in range(1, num + 1):
        _write_tif(os.path.join(image_dir, f"SN4_MS_1_{i}.tif"), count=8)

    if split == "train":
        mask_dir = os.path.join(cat_root, "geojson", "spacenet-buildings")
        os.makedirs(mask_dir, exist_ok=True)
        for i in range(1, num + 1):
            _write_geojson(
                os.path.join(mask_dir, f"SN4_geojson_1_{i}.geojson"),
                empty=(i % 2 == 1),
            )
    return str(tmp_path)


def test_getitem_train(tmp_path):
    root = _build_root(tmp_path, "train")
    ds = SpaceNet4(root=root, split="train", download=False)
    sample = ds[0]
    assert "image" in sample and "mask" in sample
    assert tuple(sample["image"].shape) == (4, 4, 8)
    assert tuple(sample["mask"].shape) == (4, 4)
    assert_dtype(sample["image"], "float32")
    assert_int64_dtype(sample["mask"])


def test_len(tmp_path):
    root = _build_root(tmp_path, "train", num=2)
    ds = SpaceNet4(root=root, split="train", download=False)
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SpaceNet4(root=str(tmp_path), split="train", download=False)
