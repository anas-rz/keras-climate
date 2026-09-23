import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.millionaid import MillionAID


def _create_file(path):
    Z = (np.random.rand(16, 16, 3) * 255).astype("uint8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(Z).convert("RGB").save(path)


@pytest.fixture
def millionaid_root(tmp_path):
    base = str(tmp_path)
    for split in ("train", "test"):
        _create_file(
            os.path.join(base, split, "agriculture_land", "grassland", "meadow", "P01.jpg")
        )
        _create_file(os.path.join(base, split, "water_area", "beach", "P02.jpg"))
    return base


def test_millionaid_getitem_multiclass(millionaid_root):
    ds = MillionAID(root=millionaid_root, task="multi-class", split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert "label" in sample
    assert tuple(sample["label"].shape) == (1,)


def test_millionaid_getitem_multilabel(millionaid_root):
    ds = MillionAID(root=millionaid_root, task="multi-label", split="train")
    # The last file has a subcategory directory (agriculture_land/grassland/meadow),
    # so its multi-label vector has 3 entries: [scene, subcategory, class].
    sample = ds[-1]
    assert tuple(sample["label"].shape) == (3,)


def test_millionaid_len(millionaid_root):
    ds = MillionAID(root=millionaid_root, split="train")
    assert len(ds) == 2


def test_millionaid_invalid_task(millionaid_root):
    with pytest.raises(AssertionError):
        MillionAID(root=millionaid_root, task="bogus")


def test_millionaid_invalid_split(millionaid_root):
    with pytest.raises(AssertionError):
        MillionAID(root=millionaid_root, split="bogus")


def test_millionaid_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        MillionAID(root=str(tmp_path), split="train")


def test_millionaid_plot(millionaid_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = MillionAID(root=millionaid_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()
