import os

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.cdl import CDL

SIZE = 8


def _write_tile(path):
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "crs": "epsg:32616",
        "transform": Affine(30, 0.0, 399960.0, 0.0, -30, 4500000.0),
        "height": SIZE,
        "width": SIZE,
    }
    data = np.random.randint(low=0, high=8, size=(SIZE, SIZE)).astype("uint8")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    _write_tile(os.path.join(root, "2023_30m_cdls.tif"))
    _write_tile(os.path.join(root, "2022_30m_cdls.tif"))
    return root


def test_getitem(prepared_root):
    ds = CDL(prepared_root, years=[2023, 2022])
    x = ds[ds.bounds]
    assert "mask" in x


def test_len(prepared_root):
    ds = CDL(prepared_root, years=[2023, 2022])
    assert len(ds) == 2


def test_classes(prepared_root):
    classes = list(CDL.valid_classes)[:5]
    ds = CDL(prepared_root, years=[2023], classes=classes)
    sample = ds[ds.bounds]
    mask = np.asarray(sample["mask"])
    assert mask.max() < len(classes)


def test_and(prepared_root):
    ds = CDL(prepared_root, years=[2023])
    combo = ds & ds
    assert isinstance(combo, IntersectionDataset)


def test_or(prepared_root):
    ds = CDL(prepared_root, years=[2023])
    combo = ds | ds
    assert isinstance(combo, UnionDataset)


def test_invalid_year(tmp_path):
    with pytest.raises(AssertionError, match="CDL data product only exists"):
        CDL(str(tmp_path), years=[1996])


def test_invalid_classes():
    with pytest.raises(AssertionError):
        CDL(classes=[-1])
    with pytest.raises(AssertionError):
        CDL(classes=[11])


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CDL(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = CDL(prepared_root, years=[2023])
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x, suptitle="Prediction")
    plt.close()
