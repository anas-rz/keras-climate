import os

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.mdas import MDAS


def _write_tif(path, num_bands, height=8, width=8, dtype="uint16"):
    crs = CRS.from_epsg(32632)
    transform = from_origin(0, 0, 1, 1)
    data = np.random.randint(0, 100, size=(num_bands, height, width)).astype(dtype)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def mdas_root(tmp_path):
    base = str(tmp_path)
    ds_root_name = "Augsburg_data_4_publication"
    subarea = "sub_area_1"
    subarea_formatted = "sub_area1"
    subarea_dir = os.path.join(base, ds_root_name, subarea)

    _write_tif(os.path.join(subarea_dir, f"3K_RGB_{subarea_formatted}.tif"), num_bands=3, dtype="uint8")
    _write_tif(os.path.join(subarea_dir, f"HySpex_{subarea_formatted}.tif"), num_bands=4, dtype="int16")
    _write_tif(os.path.join(subarea_dir, f"Sentinel_2_{subarea_formatted}.tif"), num_bands=13, dtype="uint16")
    return base


def test_mdas_getitem(mdas_root):
    ds = MDAS(root=mdas_root, subareas=["sub_area_1"], modalities=["3K_RGB", "HySpex", "Sentinel_2"])
    sample = ds[0]
    assert tuple(sample["3K_RGB_image"].shape) == (8, 8, 3)
    assert tuple(sample["HySpex_image"].shape) == (8, 8, 4)
    assert tuple(sample["Sentinel_2_image"].shape) == (8, 8, 13)


def test_mdas_len(mdas_root):
    ds = MDAS(root=mdas_root, subareas=["sub_area_1"], modalities=["3K_RGB"])
    assert len(ds) == 1


def test_mdas_invalid_subarea(mdas_root):
    with pytest.raises(AssertionError):
        MDAS(root=mdas_root, subareas=["bogus"])


def test_mdas_invalid_modality(mdas_root):
    with pytest.raises(AssertionError):
        MDAS(root=mdas_root, modalities=["bogus"])


def test_mdas_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        MDAS(root=str(tmp_path), download=False)


def test_mdas_plot(mdas_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = MDAS(root=mdas_root, subareas=["sub_area_1"], modalities=["3K_RGB"])
    ds.plot(ds[0], suptitle="Test")
    plt.close()
