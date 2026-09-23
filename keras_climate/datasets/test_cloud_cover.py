import os

import numpy as np
import pandas as pd
import pytest
import rasterio

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.cloud_cover import CloudCoverDetection

SIZE = 8


def _write_band(path):
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "crs": "epsg:4326",
        "transform": rasterio.transform.from_bounds(0, 0, 1, 1, SIZE, SIZE),
        "height": SIZE,
        "width": SIZE,
    }
    data = np.random.randint(0, 4096, size=(SIZE, SIZE)).astype("uint16")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture(params=["train", "test"])
def prepared_root(tmp_path, request):
    split = request.param
    subdir = "public" if split == "train" else "private"
    root = str(tmp_path)
    directory = os.path.join(root, subdir)

    chip_ids = ["chip_a", "chip_b"]
    for chip_id in chip_ids:
        feat_dir = os.path.join(directory, f"{split}_features", chip_id)
        os.makedirs(feat_dir, exist_ok=True)
        for band in CloudCoverDetection.all_bands:
            _write_band(os.path.join(feat_dir, f"{band}.tif"))

        label_dir = os.path.join(directory, f"{split}_labels")
        os.makedirs(label_dir, exist_ok=True)
        label_path = os.path.join(label_dir, f"{chip_id}.tif")
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "crs": "epsg:4326",
            "transform": rasterio.transform.from_bounds(0, 0, 1, 1, SIZE, SIZE),
            "height": SIZE,
            "width": SIZE,
        }
        with rasterio.open(label_path, "w", **profile) as dst:
            dst.write(np.random.randint(0, 2, size=(SIZE, SIZE)).astype("uint8"), 1)

    metadata = pd.DataFrame({"chip_id": chip_ids})
    metadata.to_csv(os.path.join(directory, f"{split}_metadata.csv"), index=False)

    return root, split


def test_getitem(prepared_root):
    root, split = prepared_root
    ds = CloudCoverDetection(root=root, split=split)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (SIZE, SIZE, 4)
    assert tuple(sample["mask"].shape) == (SIZE, SIZE)


def test_len(prepared_root):
    root, split = prepared_root
    ds = CloudCoverDetection(root=root, split=split)
    assert len(ds) == 2


def test_invalid_band(prepared_root):
    root, split = prepared_root
    with pytest.raises(AssertionError):
        CloudCoverDetection(root=root, split=split, bands=["B09"])


def test_invalid_split(tmp_path):
    with pytest.raises(AssertionError):
        CloudCoverDetection(root=str(tmp_path), split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CloudCoverDetection(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, split = prepared_root
    ds = CloudCoverDetection(root=root, split=split)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()
    sample["prediction"] = sample["mask"]
    ds.plot(sample, suptitle="Pred")
    plt.close()


def test_plot_rgb_missing(prepared_root):
    root, split = prepared_root
    ds = CloudCoverDetection(root=root, split=split, bands=["B08"])
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0], suptitle="Single Band")
