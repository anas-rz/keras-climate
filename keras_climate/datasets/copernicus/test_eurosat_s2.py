import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.copernicus.eurosat_s2 import CopernicusBenchEuroSATS2
from keras_climate.datasets.errors import DatasetNotFoundError


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    directory = os.path.join(root, "eurosat_s2")
    img_dir = os.path.join(directory, "all_imgs", "Residential")
    os.makedirs(img_dir, exist_ok=True)

    transform = from_origin(0, 8, 1, 1)
    data = np.random.randint(0, 10000, (13, 8, 8)).astype("uint16")
    with rasterio.open(
        os.path.join(img_dir, "Residential_2029.tif"),
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=13,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    for split in ("train", "val", "test"):
        with open(os.path.join(directory, f"eurosat-{split}.txt"), "w") as f:
            f.write("Residential_2029.jpg\n")

    return root


def test_getitem(prepared_root):
    ds = CopernicusBenchEuroSATS2(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 13)
    assert "label" in sample
    assert int(sample["label"]) == CopernicusBenchEuroSATS2.classes.index("Residential")


def test_len(prepared_root):
    ds = CopernicusBenchEuroSATS2(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_rgb_band_subset(prepared_root):
    ds = CopernicusBenchEuroSATS2(
        root=prepared_root, split="train", bands=("B04", "B03", "B02"), download=False
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchEuroSATS2(root=str(tmp_path), download=False)
