import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.spacenet.spacenet1 import SpaceNet1

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
    root = os.path.join(str(tmp_path), "SN1_buildings", split)
    band_dir = os.path.join(root, "8band")
    os.makedirs(band_dir, exist_ok=True)
    for i in range(1, num + 1):
        _write_tif(os.path.join(band_dir, f"8band_AOI_1_RIO_img{i}.tif"), count=8)

    if split == "train":
        geojson_dir = os.path.join(root, "geojson")
        os.makedirs(geojson_dir, exist_ok=True)
        for i in range(1, num + 1):
            _write_geojson(
                os.path.join(geojson_dir, f"Geo_AOI_1_RIO_img{i}.geojson"),
                empty=(i % 2 == 1),
            )
    return str(tmp_path)


def test_getitem_train(tmp_path):
    root = _build_root(tmp_path, "train")
    ds = SpaceNet1(root=root, split="train", image="8band", download=False)
    sample = ds[0]
    assert "image" in sample and "mask" in sample
    assert tuple(sample["image"].shape) == (102, 110, 8)
    assert tuple(sample["mask"].shape) == (102, 110)
    assert sample["image"].dtype == "float32"
    assert sample["mask"].dtype == "int64"


def test_len(tmp_path):
    root = _build_root(tmp_path, "train", num=2)
    ds = SpaceNet1(root=root, split="train", image="8band", download=False)
    assert len(ds) == 2


def test_test_split_no_mask(tmp_path):
    root = _build_root(tmp_path, "test", num=2)
    ds = SpaceNet1(root=root, split="test", image="8band", download=False)
    sample = ds[0]
    assert "mask" not in sample
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SpaceNet1(root=str(tmp_path), split="train", download=False)
