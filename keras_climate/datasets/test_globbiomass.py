import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.globbiomass import GlobBiomass


def _write_tif(path, value):
    transform = from_origin(0, 1, 0.01, 0.01)
    data = np.full((1, 8, 8), value, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    _write_tif(os.path.join(root, "N00E020_agb.tif"), 10.0)
    _write_tif(os.path.join(root, "N00E020_agb_err.tif"), 1.0)
    return root


def test_getitem(prepared_root):
    ds = GlobBiomass(paths=prepared_root, measurement="agb")
    sample = ds[ds.bounds]
    assert "mask" in sample
    assert tuple(sample["mask"].shape) == (8, 8, 2)


def test_len(prepared_root):
    ds = GlobBiomass(paths=prepared_root, measurement="agb")
    assert len(ds) == 1


def test_invalid_measurement(prepared_root):
    with pytest.raises(AssertionError):
        GlobBiomass(paths=prepared_root, measurement="invalid")


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        GlobBiomass(paths=str(tmp_path), measurement="agb")


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = GlobBiomass(paths=prepared_root, measurement="agb")
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()
