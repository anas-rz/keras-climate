import os

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.potsdam import Potsdam2D


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    name = Potsdam2D.splits["train"][0]

    image_dir = os.path.join(root, Potsdam2D.image_root)
    os.makedirs(image_dir, exist_ok=True)
    transform = from_origin(0, 8, 1, 1)
    data = np.random.randint(0, 255, (4, 8, 8)).astype("uint8")
    with rasterio.open(
        os.path.join(image_dir, f"{name}_RGBIR.tif"),
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=4,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    # Mask filled with the "Building" color from the colormap
    mask_arr = np.zeros((8, 8, 3), dtype="uint8")
    mask_arr[..., :] = Potsdam2D.colormap[2]
    Image.fromarray(mask_arr, mode="RGB").save(os.path.join(root, f"{name}_label.tif"))

    return root


def test_getitem(prepared_root):
    ds = Potsdam2D(root=prepared_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 4)
    assert tuple(sample["mask"].shape) == (8, 8)
    assert int(sample["mask"][0, 0]) == 2


def test_len(prepared_root):
    ds = Potsdam2D(root=prepared_root, split="train")
    assert len(ds) == 1


def test_invalid_split():
    with pytest.raises(AssertionError):
        Potsdam2D(split="foo")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        Potsdam2D(root=str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Potsdam2D(root=prepared_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()
