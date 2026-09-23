import os

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.naip import NAIP

SIZE = 8


def _write_tile(path):
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 4,
        "crs": "epsg:26918",
        "transform": Affine(1.0, 0.0, 1303555.0, 0.0, -1.0, 2535065.0),
        "height": SIZE,
        "width": SIZE,
    }
    data = np.random.randint(low=0, high=255, size=(4, SIZE, SIZE)).astype("uint8")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    _write_tile(os.path.join(root, "m_3807511_ne_18_060_20181104.tif"))
    _write_tile(os.path.join(root, "m_3807511_ne_18_060_20190605.tif"))
    return root


def test_getitem(prepared_root):
    ds = NAIP(prepared_root)
    x = ds[ds.bounds]
    assert tuple(x["image"].shape[-3:]) == (SIZE, SIZE, 4)


def test_len(prepared_root):
    ds = NAIP(prepared_root)
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        NAIP(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = NAIP(prepared_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
