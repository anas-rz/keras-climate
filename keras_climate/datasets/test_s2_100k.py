import os

import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from rasterio import Affine
from rasterio.crs import CRS

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.s2_100k import S2100k


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    images_dir = os.path.join(root, "images")
    os.makedirs(images_dir, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": 8,
        "height": 8,
        "count": 12,
        "crs": CRS.from_epsg(32639),
        "transform": Affine(10.0, 0.0, 524385.0, 0.0, -10.0, 2712815.0),
    }
    data = np.random.randint(0, 1000, (12, 8, 8)).astype("uint16")
    with rio.open(os.path.join(images_dir, "patch_0.tif"), "w", **profile) as dst:
        dst.write(data)

    df = pd.DataFrame({"fn": ["patch_0.tif"], "lon": [10.5], "lat": [45.2]})
    df.to_csv(os.path.join(root, "index.csv"), index=False)

    return root


def test_getitem_both(prepared_root):
    ds = S2100k(root=prepared_root, mode="both", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 12)
    assert tuple(sample["point"].shape) == (2,)


def test_getitem_points_only(prepared_root):
    ds = S2100k(root=prepared_root, mode="points", download=False)
    sample = ds[0]
    assert "image" not in sample
    assert "point" in sample


def test_len(prepared_root):
    ds = S2100k(root=prepared_root, download=False)
    assert len(ds) == 1


def test_invalid_mode():
    with pytest.raises(AssertionError):
        S2100k(mode="foo")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        S2100k(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = S2100k(root=prepared_root, download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
