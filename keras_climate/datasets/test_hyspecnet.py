import os

import numpy as np
import pytest
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.hyspecnet import HySpecNet11k

SIZE = 8
DTYPE = "int16"

TILE = "ENMAP01-____L2A-DT0000004950_20221103T162438Z_001_V010110_20221118T145147Z"
PATCH = "Y01460273_X05670694"


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    profile = {
        "driver": "GTiff",
        "dtype": DTYPE,
        "nodata": -32768.0,
        "width": SIZE,
        "height": SIZE,
        "count": 224,
        "crs": CRS.from_epsg(32618),
        "transform": Affine(30.0, 0.0, 691845.0, 0.0, -30.0, 4561935.0),
    }

    patch_stub = f"{TILE}-{PATCH}"
    for strategy in ("easy",):
        split_dir = os.path.join(root, "hyspecnet-11k", "splits", strategy)
        os.makedirs(split_dir, exist_ok=True)
        for split in ("train", "val", "test"):
            with open(os.path.join(split_dir, f"{split}.csv"), "w") as f:
                f.write(f"{patch_stub}-DATA.npy\n")

    patches_dir = os.path.join(root, "hyspecnet-11k", "patches")
    os.makedirs(patches_dir, exist_ok=True)
    tif_path = os.path.join(patches_dir, f"{patch_stub}-SPECTRAL_IMAGE.TIF")
    data = np.random.randint(-100, 100, size=(SIZE, SIZE)).astype(DTYPE)
    with rasterio.open(tif_path, "w", **profile) as dst:
        for i in range(1, profile["count"] + 1):
            dst.write(data, i)

    return root


def test_getitem(prepared_root):
    ds = HySpecNet11k(root=prepared_root, split="train", strategy="easy")
    sample = ds[0]
    assert "image" in sample
    assert tuple(sample["image"].shape) == (SIZE, SIZE, len(ds.bands))
    assert "wavelength" in sample


def test_len(prepared_root):
    ds = HySpecNet11k(root=prepared_root, split="train", strategy="easy")
    assert len(ds) == 1


def test_band_subset(prepared_root):
    ds = HySpecNet11k(
        root=prepared_root, split="train", strategy="easy", bands=("B48", "B30", "B16")
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (SIZE, SIZE, 3)


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        HySpecNet11k(root=str(tmp_path), split="train", download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = HySpecNet11k(root=prepared_root, split="train", strategy="easy")
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()


def test_plot_rgb_missing_band_raises(prepared_root):
    from keras_climate.datasets import RGBBandsMissingError

    ds = HySpecNet11k(
        root=prepared_root, split="train", strategy="easy", bands=("B48",)
    )
    sample = ds[0]
    with pytest.raises(RGBBandsMissingError):
        ds.plot(sample)
