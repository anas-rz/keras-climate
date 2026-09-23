import os

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.nccm import NCCM

SIZE = 8


def _write_tile(path):
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "crs": "epsg:32616",
        "transform": Affine(10, 0.0, 399960.0, 0.0, -10, 4500000.0),
        "height": SIZE,
        "width": SIZE,
    }
    data = np.random.choice([0, 1, 2, 3, 15], size=(SIZE, SIZE)).astype("uint8")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    _write_tile(os.path.join(root, "CDL2019_clip.tif"))
    _write_tile(os.path.join(root, "CDL2018_clip1.tif"))
    return root


def test_getitem(prepared_root):
    ds = NCCM(prepared_root, years=[2019, 2018])
    x = ds[ds.bounds]
    assert "mask" in x
    mask = np.asarray(x["mask"])
    # nodata class 15 should have been remapped to 4
    assert mask.max() <= 4


def test_len(prepared_root):
    ds = NCCM(prepared_root, years=[2019, 2018])
    assert len(ds) == 2


def test_invalid_year(tmp_path):
    with pytest.raises(AssertionError, match="NCCM data product only exists"):
        NCCM(str(tmp_path), years=[1999])


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        NCCM(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = NCCM(prepared_root, years=[2019])
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x, suptitle="Prediction")
    plt.close()
