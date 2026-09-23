import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.gse import GoogleSatelliteEmbedding


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    year_dir = os.path.join(root, "2024")
    os.makedirs(year_dir, exist_ok=True)
    path = os.path.join(year_dir, "tile.tif")
    transform = from_origin(0, 1, 0.01, 0.01)
    data = np.random.rand(64, 8, 8).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=64,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return root


def test_getitem(prepared_root):
    ds = GoogleSatelliteEmbedding(paths=prepared_root)
    sample = ds[ds.bounds]
    assert "image" in sample
    assert tuple(sample["image"].shape) == (8, 8, 64)


def test_len(prepared_root):
    ds = GoogleSatelliteEmbedding(paths=prepared_root)
    assert len(ds) == 1


def test_all_bands():
    assert len(GoogleSatelliteEmbedding.all_bands) == 64
    assert GoogleSatelliteEmbedding.all_bands[0] == "A00"
    assert GoogleSatelliteEmbedding.all_bands[-1] == "A63"


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = GoogleSatelliteEmbedding(paths=prepared_root)
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()
