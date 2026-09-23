import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.eudem import EUDEM


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    path = os.path.join(root, "eu_dem_v11_E30N10.TIF")
    transform = from_origin(0, 8, 1, 1)
    data = np.random.randint(0, 1000, (1, 8, 8)).astype("int16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return root


def test_getitem(prepared_root):
    ds = EUDEM(paths=prepared_root)
    sample = ds[ds.bounds]
    assert "mask" in sample
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(prepared_root):
    ds = EUDEM(paths=prepared_root)
    assert len(ds) == 1


def test_and_or(prepared_root):
    ds = EUDEM(paths=prepared_root)
    assert isinstance(ds & ds, IntersectionDataset)
    assert isinstance(ds | ds, UnionDataset)


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EUDEM(paths=str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = EUDEM(paths=prepared_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x)
    plt.close()
