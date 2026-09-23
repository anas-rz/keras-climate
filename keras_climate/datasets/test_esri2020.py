import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.esri2020 import Esri2020


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    path = os.path.join(root, "17S_20200101-20210101.tif")
    transform = from_origin(0, 8, 1, 1)
    data = np.random.randint(0, 11, (1, 8, 8)).astype("uint8")
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
    return root


def test_getitem(prepared_root):
    ds = Esri2020(paths=prepared_root, download=False)
    sample = ds[ds.bounds]
    assert "mask" in sample
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(prepared_root):
    ds = Esri2020(paths=prepared_root, download=False)
    assert len(ds) == 1


def test_and_or(prepared_root):
    ds = Esri2020(paths=prepared_root, download=False)
    assert isinstance(ds & ds, IntersectionDataset)
    assert isinstance(ds | ds, UnionDataset)


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        Esri2020(paths=str(tmp_path), download=False)


def test_url():
    assert "ai4edataeuwest.blob.core.windows.net" in Esri2020.url


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Esri2020(paths=prepared_root, download=False)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x)
    plt.close()
