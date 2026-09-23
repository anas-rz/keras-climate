import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.spacenet.spacenet6 import SpaceNet6

TRANSFORM = from_origin(0, 4, 1, 1)
CRS = "EPSG:4326"
AOI = 11

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
    aoi_root = os.path.join(str(tmp_path), "SN6_buildings", split, f"AOI_{AOI}_Rotterdam")
    image_name = "PAN" if split == "train" else "SAR-Intensity"
    image_dir = os.path.join(aoi_root, image_name)
    os.makedirs(image_dir, exist_ok=True)
    for i in range(1, num + 1):
        _write_tif(
            os.path.join(image_dir, f"{image_name}_AOI_{AOI}_Rotterdam_tile_{i}.tif"),
            count=4,
        )

    if split == "train":
        mask_dir = os.path.join(aoi_root, "geojson_buildings")
        os.makedirs(mask_dir, exist_ok=True)
        for i in range(1, num + 1):
            _write_geojson(
                os.path.join(
                    mask_dir, f"Buildings_AOI_{AOI}_Rotterdam_tile_{i}.geojson"
                ),
                empty=(i % 2 == 1),
            )
    return str(tmp_path)


def test_getitem_train(tmp_path):
    root = _build_root(tmp_path, "train")
    ds = SpaceNet6(root=root, split="train", aois=[AOI], download=False)
    sample = ds[0]
    assert "image" in sample and "mask" in sample
    assert tuple(sample["image"].shape) == (4, 4, 4)
    assert tuple(sample["mask"].shape) == (4, 4)
    assert sample["image"].dtype == "float32"
    assert sample["mask"].dtype == "int64"


def test_len(tmp_path):
    root = _build_root(tmp_path, "train", num=2)
    ds = SpaceNet6(root=root, split="train", aois=[AOI], download=False)
    assert len(ds) == 2


def test_getitem_test_split(tmp_path):
    root = _build_root(tmp_path, "test", num=2)
    ds = SpaceNet6(root=root, split="test", aois=[AOI], download=False)
    sample = ds[0]
    assert "mask" not in sample
    assert tuple(sample["image"].shape) == (4, 4, 4)
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SpaceNet6(root=str(tmp_path), split="train", aois=[AOI], download=False)
