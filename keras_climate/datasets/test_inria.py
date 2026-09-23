import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.inria import InriaAerialImageLabeling


def _write_tif(path, count, value, dtype):
    transform = from_origin(0, 1, 0.1, 0.1)
    data = np.full((count, 8, 8), value, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    for city_prefix, idx in [("austin", 1), ("austin", 6)]:
        img_dir = os.path.join(root, "AerialImageDataset", "train", "images")
        gt_dir = os.path.join(root, "AerialImageDataset", "train", "gt")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        fname = f"{city_prefix}{idx}.tif"
        _write_tif(os.path.join(img_dir, fname), 3, 100, "uint8")
        _write_tif(os.path.join(gt_dir, fname), 1, 1, "uint8")

    test_img_dir = os.path.join(root, "AerialImageDataset", "test", "images")
    os.makedirs(test_img_dir, exist_ok=True)
    _write_tif(os.path.join(test_img_dir, "bellingham1.tif"), 3, 100, "uint8")

    return root


def test_train_getitem(prepared_root):
    ds = InriaAerialImageLabeling(root=prepared_root, split="train")
    assert len(ds) == 1
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_val_getitem(prepared_root):
    ds = InriaAerialImageLabeling(root=prepared_root, split="val")
    assert len(ds) == 1
    sample = ds[0]
    assert "mask" in sample


def test_test_getitem(prepared_root):
    ds = InriaAerialImageLabeling(root=prepared_root, split="test")
    assert len(ds) == 1
    sample = ds[0]
    assert "image" in sample
    assert "mask" not in sample


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        InriaAerialImageLabeling(root=prepared_root, split="bogus")


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        InriaAerialImageLabeling(root=str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = InriaAerialImageLabeling(root=prepared_root, split="train")
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()
