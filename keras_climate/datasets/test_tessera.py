import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.tessera import TesseraEmbeddings


@pytest.fixture
def prepared_root(tmp_path):
    transform = from_origin(0, 8, 1, 1)
    path = os.path.join(str(tmp_path), "grid_1.0_2.0_2021.tiff")
    data = np.random.rand(128, 8, 8).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=128,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = TesseraEmbeddings(paths=prepared_root)
    sample = ds[0:8, 0:8]
    assert tuple(sample["image"].shape) == (8, 8, 128)


def test_len(prepared_root):
    ds = TesseraEmbeddings(paths=prepared_root)
    assert len(ds) == 1


def test_all_bands():
    assert len(TesseraEmbeddings.all_bands) == 128


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = TesseraEmbeddings(paths=prepared_root)
    sample = ds[0:8, 0:8]
    ds.plot(sample, suptitle="Test")
    plt.close()
