import os
import shutil

import numpy as np
import pytest
import rasterio

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.cms_mangrove_canopy import CMSGlobalMangroveCanopy

SIZE = 8


def _write_tile(path):
    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "count": 1,
        "crs": "epsg:4326",
        "transform": rasterio.transform.from_bounds(0, 0, 1, 1, SIZE, SIZE),
        "height": SIZE,
        "width": SIZE,
    }
    data = np.random.randint(0, 100, size=(1, SIZE, SIZE)).astype("int32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    directory = os.path.join(root, "CMS_Global_Map_Mangrove_Canopy_1665", "data")
    os.makedirs(directory, exist_ok=True)
    _write_tile(os.path.join(directory, "Mangrove_agb_Angola.tif"))

    # _verify() also requires the zip to be present
    zip_base = os.path.join(root, "CMS_Global_Map_Mangrove_Canopy_1665")
    shutil.make_archive(zip_base, "zip", root, "CMS_Global_Map_Mangrove_Canopy_1665")

    return root


def test_getitem(prepared_root):
    ds = CMSGlobalMangroveCanopy(prepared_root, measurement="agb", country="Angola", checksum=False)
    x = ds[ds.bounds]
    assert "mask" in x


def test_len(prepared_root):
    ds = CMSGlobalMangroveCanopy(prepared_root, measurement="agb", country="Angola", checksum=False)
    assert len(ds) == 1


def test_and(prepared_root):
    ds = CMSGlobalMangroveCanopy(prepared_root, measurement="agb", country="Angola", checksum=False)
    assert isinstance(ds & ds, IntersectionDataset)


def test_or(prepared_root):
    ds = CMSGlobalMangroveCanopy(prepared_root, measurement="agb", country="Angola", checksum=False)
    assert isinstance(ds | ds, UnionDataset)


def test_invalid_country(tmp_path):
    with pytest.raises(AssertionError):
        CMSGlobalMangroveCanopy(str(tmp_path), country="Nowhere")


def test_invalid_measurement(tmp_path):
    with pytest.raises(AssertionError):
        CMSGlobalMangroveCanopy(str(tmp_path), measurement="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CMSGlobalMangroveCanopy(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = CMSGlobalMangroveCanopy(prepared_root, measurement="agb", country="Angola", checksum=False)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x, suptitle="Prediction")
    plt.close()
