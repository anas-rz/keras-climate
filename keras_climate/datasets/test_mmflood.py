import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.mmflood import MMFlood

HEIGHT = WIDTH = 16

PROFILE_BASE = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": None,
    "crs": CRS.from_epsg(4326),
    "transform": Affine(
        0.0001287974837883981,
        0.0,
        14.438064999669106,
        0.0,
        -8.989523639880024e-05,
        45.71617928533084,
    ),
    "height": HEIGHT,
    "width": WIDTH,
}


def _generate_data(path, filename, include_hydro=False):
    folders_data = {
        "s1_raw": np.random.rand(2, HEIGHT, WIDTH).astype("float32"),
        "DEM": np.random.rand(1, HEIGHT, WIDTH).astype("float32"),
        "mask": np.random.randint(0, 2, size=(1, HEIGHT, WIDTH)).astype("uint8"),
    }
    if include_hydro:
        folders_data["hydro"] = np.random.rand(1, HEIGHT, WIDTH).astype("float32")

    for folder, data in folders_data.items():
        folder_path = os.path.join(path, folder)
        os.makedirs(folder_path, exist_ok=True)
        filepath = os.path.join(folder_path, filename)
        profile = dict(PROFILE_BASE)
        profile["count"] = data.shape[0]
        with rasterio.open(filepath, "w", **profile) as dst:
            dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    datapath = os.path.join(root, "activations")

    folders_splits = [
        ("EMSR000", "train"),
        ("EMSR001", "train"),
        ("EMSR003", "val"),
        ("EMSR004", "test"),
    ]
    num_files = {"EMSR000": 3, "EMSR001": 2, "EMSR003": 2, "EMSR004": 1}
    num_hydro = {"EMSR001": 2, "EMSR003": 1, "EMSR004": 1}

    metadata = {}
    for folder, split in folders_splits:
        dst_folder = os.path.join(datapath, f"{folder}-0")
        count_hydro = 0
        for idx in range(num_files[folder]):
            include_hydro = count_hydro < num_hydro.get(folder, 0)
            _generate_data(dst_folder, f"{folder}-{idx}.tif", include_hydro=include_hydro)
            if include_hydro:
                count_hydro += 1

        metadata[folder] = {
            "title": "Test flood",
            "type": "Flood",
            "country": "N/A",
            "start": "2014-11-06T17:57:00",
            "end": "2015-01-29T12:47:04",
            "lat": 45.82427031690563,
            "lon": 14.484407562009336,
            "subset": split,
            "delineations": [f"{folder}_00"],
        }

    with open(os.path.join(root, "activations.json"), "w") as fp:
        json.dump(metadata, fp)

    return root


def test_getitem(prepared_root):
    ds = MMFlood(prepared_root, split="train")
    x = ds[ds.bounds]
    assert "image" in x
    assert "mask" in x
    assert x["image"].shape[-1] == 2


def test_getitem_with_dem_and_hydro(prepared_root):
    ds = MMFlood(prepared_root, split="train", include_dem=True, include_hydro=True)
    x = ds[ds.bounds]
    assert x["image"].shape[-1] == 4


def test_len_train(prepared_root):
    ds = MMFlood(prepared_root, split="train")
    assert len(ds) == 5


def test_len_train_hydro(prepared_root):
    ds = MMFlood(prepared_root, split="train", include_hydro=True)
    assert len(ds) == 2


def test_len_val(prepared_root):
    ds = MMFlood(prepared_root, split="val")
    assert len(ds) == 2


def test_len_test(prepared_root):
    ds = MMFlood(prepared_root, split="test")
    assert len(ds) == 1


def test_and(prepared_root):
    ds = MMFlood(prepared_root, split="train")
    combo = ds & ds
    assert isinstance(combo, IntersectionDataset)


def test_or(prepared_root):
    ds = MMFlood(prepared_root, split="train")
    combo = ds | ds
    assert isinstance(combo, UnionDataset)


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        MMFlood(prepared_root, split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        MMFlood(str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = MMFlood(prepared_root, split="train", include_dem=True, include_hydro=True)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x, show_titles=False)
    plt.close()
