import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.oscd import OSCD, OSCD100

SIZE = 16


def _write_region(images_root, labels_root, region, bands):
    for rect in ("imgs_1_rect", "imgs_2_rect"):
        d = os.path.join(images_root, region, rect)
        os.makedirs(d, exist_ok=True)
        for band in bands:
            arr = np.random.randint(0, 65535, size=(SIZE, SIZE), dtype="uint16")
            Image.fromarray(arr).save(os.path.join(d, f"{band}.tif"))

    with open(os.path.join(images_root, region, "dates.txt"), "w") as f:
        f.write("date_1: 20161130\ndate_2: 20170829\n")

    mask_dir = os.path.join(labels_root, region, "cm")
    os.makedirs(mask_dir, exist_ok=True)
    mask_arr = np.random.randint(0, 255, size=(SIZE, SIZE), dtype="uint8")
    Image.fromarray(mask_arr).save(os.path.join(mask_dir, "cm.png"))


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    images_root = os.path.join(root, "Onera Satellite Change Detection dataset - Images")
    labels_root = os.path.join(root, "Onera Satellite Change Detection dataset - Train Labels")
    _write_region(images_root, labels_root, "region1", OSCD.all_bands)
    return root


def test_getitem(prepared_root):
    ds = OSCD(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, SIZE, SIZE, 13)
    assert tuple(sample["mask"].shape) == (1, SIZE, SIZE)


def test_len(prepared_root):
    ds = OSCD(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_band_subset(prepared_root):
    ds = OSCD(root=prepared_root, split="train", bands=("B04", "B03", "B02"), download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, SIZE, SIZE, 3)


def test_invalid_split():
    with pytest.raises(AssertionError):
        OSCD(split="bogus")


def test_invalid_bands():
    with pytest.raises(AssertionError):
        OSCD(bands=("BOGUS",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        OSCD(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = OSCD(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()


def test_plot_missing_rgb(prepared_root):
    ds = OSCD(root=prepared_root, split="train", bands=("B01",), download=False)
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_oscd100_splits():
    assert OSCD100.splits == ("train", "val", "test")
