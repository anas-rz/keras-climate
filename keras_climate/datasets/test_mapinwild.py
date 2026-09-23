import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.mapinwild import MapInWild


def _write_tif(path, count, height=8, width=8, dtype="uint16"):
    crs = CRS.from_epsg(32720)
    transform = from_origin(0, 8, 1, 1)
    data = np.random.randint(0, 100, size=(count, height, width)).astype(dtype)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def mapinwild_root(tmp_path):
    base = str(tmp_path)
    for source, count, dtype in [("mask", 1, "uint8"), ("ESA_WC", 1, "uint8"), ("VIIRS", 1, "float32")]:
        _write_tif(os.path.join(base, source, "0.tif"), count=count, dtype=dtype)

    pd.DataFrame({"train": [0], "validation": [np.nan], "test": [np.nan]}).to_csv(
        os.path.join(base, "split_IDs.csv"), index=False
    )
    return base


def test_mapinwild_getitem(mapinwild_root):
    ds = MapInWild(root=mapinwild_root, modality=["esa_wc", "viirs"], split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 2)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_mapinwild_len(mapinwild_root):
    ds = MapInWild(root=mapinwild_root, modality=["esa_wc"], split="train")
    assert len(ds) == 1


def test_mapinwild_invalid_split(mapinwild_root):
    with pytest.raises(AssertionError):
        MapInWild(root=mapinwild_root, modality=["esa_wc"], split="bogus")


def test_mapinwild_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        MapInWild(root=str(tmp_path), modality=["esa_wc"], split="train", download=False)


def test_mapinwild_plot(mapinwild_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = MapInWild(root=mapinwild_root, modality=["esa_wc"], split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()
