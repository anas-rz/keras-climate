import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.south_america_soybean import SouthAmericaSoybean


@pytest.fixture
def prepared_root(tmp_path):
    transform = from_origin(0, 8, 1, 1)
    path = os.path.join(str(tmp_path), "SouthAmerica_Soybean_2021.tif")
    data = np.random.randint(0, 2, (1, 8, 8)).astype("uint8")
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
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = SouthAmericaSoybean(paths=prepared_root, years=[2021])
    sample = ds[0:8, 0:8]
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(prepared_root):
    ds = SouthAmericaSoybean(paths=prepared_root, years=[2021])
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SouthAmericaSoybean(paths=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SouthAmericaSoybean(paths=prepared_root, years=[2021])
    sample = ds[0:8, 0:8]
    ds.plot(sample, suptitle="Test")
    plt.close()
