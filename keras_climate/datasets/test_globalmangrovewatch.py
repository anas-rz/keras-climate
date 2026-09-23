import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.globalmangrovewatch import GlobalMangroveWatch


def _write_tif(path, year, value):
    transform = from_origin(0, 1, 0.01, 0.01)
    data = np.full((1, 8, 8), value, dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    _write_tif(os.path.join(root, "GMW_N10E010_2020_v3.tif"), 2020, 1)
    _write_tif(os.path.join(root, "GMW_N10E011_2019_v3.tif"), 2019, 0)
    return root


def test_getitem_and_len(prepared_root):
    ds = GlobalMangroveWatch(paths=prepared_root, years=(2020,))
    assert len(ds) == 1
    sample = ds[ds.bounds]
    assert "mask" in sample
    assert tuple(sample["mask"].shape) == (8, 8)


def test_multiple_years(prepared_root):
    ds = GlobalMangroveWatch(paths=prepared_root, years=(2020, 2019))
    assert len(ds) == 2


def test_invalid_years(prepared_root):
    with pytest.raises(AssertionError):
        GlobalMangroveWatch(paths=prepared_root, years=(1999,))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        GlobalMangroveWatch(paths=str(tmp_path), years=(2020,), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = GlobalMangroveWatch(paths=prepared_root, years=(2020,))
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()
