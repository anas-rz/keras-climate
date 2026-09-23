import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

from keras_climate.datasets.copernicus.lc100seg_s3 import CopernicusBenchLC100SegS3
from keras_climate.datasets.errors import DatasetNotFoundError

LOCATION = "0200599_-70.25_-55.25"
FILES = [
    "S3A_20190507T135736_20190507T135744.tif",
    "S3B_20190810T135525_20190810T135539.tif",
]
CLASS_VALUES = [
    0,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    111,
    112,
    113,
    114,
    115,
    116,
    121,
    122,
    123,
    124,
    125,
    126,
    200,
]


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    directory = os.path.join(root, "lc100_s3")
    img_dir = os.path.join(directory, "s3_olci", LOCATION)
    mask_dir = os.path.join(directory, "lc100")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    size = 9
    profile = {
        "driver": "GTiff",
        "crs": CRS.from_epsg(4326),
        "width": size,
        "height": size,
        "count": 21,
        "dtype": "float32",
        "transform": Affine(0.0027, 0.0, -69.62, 0.0, -0.0027, -55.37),
    }
    data = np.random.random(size=(size, size)).astype("float32")
    for file in FILES:
        with rasterio.open(os.path.join(img_dir, file), "w", **profile) as dst:
            for i in range(1, profile["count"] + 1):
                dst.write(data, i)

    mask_profile = dict(profile)
    mask_profile["count"] = 1
    mask_profile["dtype"] = "uint8"
    mask = np.random.choice(CLASS_VALUES, size=(size, size)).astype("uint8")
    with rasterio.open(
        os.path.join(mask_dir, f"{LOCATION}.tif"), "w", **mask_profile
    ) as dst:
        dst.write(mask, 1)

    for split in ("train", "val", "test"):
        df = pd.DataFrame([[LOCATION, *list(np.random.randint(0, 2, size=(23,)))]])
        df.to_csv(os.path.join(directory, f"multilabel-{split}.csv"), index=None)
        df = pd.DataFrame([[LOCATION, FILES[0]]])
        df.to_csv(
            os.path.join(directory, f"static_fnames-{split}.csv"),
            header=None,
            index=None,
        )

    return root


def test_getitem_static(prepared_root):
    ds = CopernicusBenchLC100SegS3(
        root=prepared_root, split="train", mode="static", download=False
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (9, 9, 21)
    assert tuple(sample["mask"].shape) == (9, 9)
    mask = np.array(sample["mask"])
    assert mask.max() <= 22
    assert mask.min() >= 0


def test_getitem_time_series(prepared_root):
    ds = CopernicusBenchLC100SegS3(
        root=prepared_root, split="train", mode="time-series", download=False
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, 9, 9, 21)
    assert tuple(sample["mask"].shape) == (9, 9)


def test_len(prepared_root):
    ds = CopernicusBenchLC100SegS3(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchLC100SegS3(root=str(tmp_path), download=False)
