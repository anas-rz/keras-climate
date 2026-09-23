import os

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.nlcd import NLCD

SIZE = 8


def _write_tile(path):
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "crs": "epsg:5070",
        "transform": Affine(30, 0.0, -2493045.0, 0.0, -30, 3310005.0),
        "height": SIZE,
        "width": SIZE,
    }
    allowed = [0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95, 250]
    data = np.random.choice(allowed, size=(SIZE, SIZE)).astype("uint8")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    _write_tile(os.path.join(root, "Annual_NLCD_LndCov_2023_CU_C1V1.tif"))
    _write_tile(os.path.join(root, "Annual_NLCD_LndCov_2022_CU_C1V1.tif"))
    return root


def test_getitem(prepared_root):
    ds = NLCD(prepared_root, years=[2023, 2022])
    x = ds[ds.bounds]
    assert "mask" in x


def test_len(prepared_root):
    ds = NLCD(prepared_root, years=[2023, 2022])
    assert len(ds) == 2


def test_classes(prepared_root):
    classes = list(NLCD.valid_classes)[:5] + [250]
    ds = NLCD(prepared_root, years=[2023], classes=classes)
    sample = ds[ds.bounds]
    mask = np.asarray(sample["mask"])
    assert mask.max() < len(classes)


def test_and(prepared_root):
    ds = NLCD(prepared_root, years=[2023])
    combo = ds & ds
    assert isinstance(combo, IntersectionDataset)


def test_or(prepared_root):
    ds = NLCD(prepared_root, years=[2023])
    combo = ds | ds
    assert isinstance(combo, UnionDataset)


def test_invalid_year(tmp_path):
    with pytest.raises(AssertionError, match="NLCD data product only exists"):
        NLCD(str(tmp_path), years=[1900])


def test_invalid_classes():
    with pytest.raises(AssertionError):
        NLCD(classes=[-1])
    with pytest.raises(AssertionError):
        NLCD(classes=[11])


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        NLCD(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = NLCD(prepared_root, years=[2023])
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x, suptitle="Prediction")
    plt.close()
