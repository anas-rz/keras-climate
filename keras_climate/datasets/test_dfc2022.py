import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.dfc2022 import DFC2022


def _write_tif(path, dtype, count, size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": count,
        "crs": "epsg:4326",
        "transform": from_bounds(0, 0, 1, 1, size, size),
        "height": size,
        "width": size,
    }
    if "float" in dtype:
        data = np.random.randn(size, size).astype(dtype)
    else:
        data = np.random.randint(0, 255, (size, size)).astype(dtype)
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(1, count + 1):
            dst.write(data, i)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)

    for split_name, info in DFC2022.metadata.items():
        directory = os.path.join(root, info["directory"])
        region = "Region1"
        image_path = os.path.join(directory, region, "BDORTHO", "tile_1.tif")
        dem_path = os.path.join(directory, region, "RGEALTI", "tile_1_RGEALTI.tif")
        _write_tif(image_path, "uint8", 3)
        _write_tif(dem_path, "float32", 1)
        if split_name == "train":
            target_path = os.path.join(directory, region, "UrbanAtlas", "tile_1_UA2012.tif")
            _write_tif(target_path, "uint8", 1)

    return root


def test_getitem_train(prepared_root):
    ds = DFC2022(root=prepared_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 4)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_getitem_val(prepared_root):
    ds = DFC2022(root=prepared_root, split="val")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 4)
    assert "mask" not in sample


def test_len(prepared_root):
    ds = DFC2022(root=prepared_root, split="train")
    assert len(ds) == 1


def test_invalid_split():
    with pytest.raises(AssertionError):
        DFC2022(split="bad")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        DFC2022(root=str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DFC2022(root=prepared_root, split="train")
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()

    sample["prediction"] = sample["mask"]
    ds.plot(sample)
    plt.close()
