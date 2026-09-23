import os

import numpy as np
import pytest
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

from keras_climate.datasets.copernicus.embed import CopernicusEmbed
from keras_climate.datasets.errors import DatasetNotFoundError


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    size = 8
    count = 5

    np.random.seed(0)
    data = np.random.random(size=(count, size, size)).astype("float32") * 2 - 1
    # Zero out a corner pixel to exercise the "invalid" (all-zero) pixel path
    data[:, 0, 0] = 0

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": size,
        "height": size,
        "count": count,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(0.25, 0.0, -180.125, 0.0, -0.25, 90.125),
    }
    with rasterio.open(os.path.join(root, "embed_map_test.tif"), "w", **profile) as dst:
        dst.write(data)

    return root


def test_getitem(prepared_root):
    ds = CopernicusEmbed(paths=prepared_root, download=False)
    sample = ds[ds.bounds]
    assert tuple(sample["image"].shape) == (8, 8, 5)


def test_len(prepared_root):
    ds = CopernicusEmbed(paths=prepared_root, download=False)
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusEmbed(paths=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = CopernicusEmbed(paths=prepared_root, download=False)
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()
