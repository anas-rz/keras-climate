import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.copernicus.eurosat_s1 import CopernicusBenchEuroSATS1
from keras_climate.datasets.errors import DatasetNotFoundError


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    directory = os.path.join(root, "eurosat_s1")
    img_dir = os.path.join(directory, "all_imgs", "Residential")
    os.makedirs(img_dir, exist_ok=True)

    transform = from_origin(0, 8, 1, 1)
    data = np.random.rand(2, 8, 8).astype("float32")
    with rasterio.open(
        os.path.join(img_dir, "Residential_2029.tif"),
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    for split in ("train", "val", "test"):
        with open(os.path.join(directory, f"eurosat-{split}.txt"), "w") as f:
            f.write("Residential_2029.jpg\n")

    return root


def test_getitem(prepared_root):
    ds = CopernicusBenchEuroSATS1(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 2)
    assert "label" in sample
    assert int(sample["label"]) == CopernicusBenchEuroSATS1.classes.index("Residential")


def test_len(prepared_root):
    ds = CopernicusBenchEuroSATS1(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchEuroSATS1(root=str(tmp_path), download=False)
