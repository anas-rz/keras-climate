import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.ftw import FieldsOfTheWorld


def _write_tif(path, count, dtype, transform):
    data = np.random.randint(0, 255, (count, 8, 8)).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    country = "austria"
    country_dir = os.path.join(root, country)

    win_a_dir = os.path.join(country_dir, "s2_images", "window_a")
    win_b_dir = os.path.join(country_dir, "s2_images", "window_b")
    mask_dir = os.path.join(country_dir, "label_masks", "semantic_2class")
    os.makedirs(win_a_dir, exist_ok=True)
    os.makedirs(win_b_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    transform = from_origin(0, 8, 1, 1)
    _write_tif(os.path.join(win_a_dir, "aoi1.tif"), 4, "uint8", transform)
    _write_tif(os.path.join(win_b_dir, "aoi1.tif"), 4, "uint8", transform)
    _write_tif(os.path.join(mask_dir, "aoi1.tif"), 1, "uint8", transform)

    df = pd.DataFrame({"aoi_id": ["aoi1"], "split": ["train"]})
    df.to_parquet(os.path.join(country_dir, f"chips_{country}.parquet"))

    return root


def test_getitem(prepared_root):
    ds = FieldsOfTheWorld(root=prepared_root, split="train", countries="austria", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 8)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(prepared_root):
    ds = FieldsOfTheWorld(root=prepared_root, split="train", countries="austria", download=False)
    assert len(ds) == 1


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        FieldsOfTheWorld(root=prepared_root, split="bogus", countries="austria")


def test_invalid_country(prepared_root):
    with pytest.raises(AssertionError):
        FieldsOfTheWorld(root=prepared_root, countries="nowhere")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        FieldsOfTheWorld(root=str(tmp_path), countries="austria", download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = FieldsOfTheWorld(root=prepared_root, split="train", countries="austria", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
