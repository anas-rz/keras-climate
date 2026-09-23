import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.loveda import LoveDA


@pytest.fixture
def loveda_root(tmp_path):
    base = str(tmp_path)
    for split in ("Train", "Val", "Test"):
        for scene in ("Rural", "Urban"):
            img_dir = os.path.join(base, split, scene, "images_png")
            mask_dir = os.path.join(base, split, scene, "masks_png")
            os.makedirs(img_dir)
            os.makedirs(mask_dir)
            img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(img_dir, "01.png"))
            mask = np.random.randint(0, 7, (16, 16), dtype=np.uint8)
            Image.fromarray(mask).save(os.path.join(mask_dir, "01.png"))
    return base


def test_loveda_getitem_train(loveda_root):
    ds = LoveDA(root=loveda_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert tuple(sample["mask"].shape) == (16, 16)


def test_loveda_getitem_test_no_mask(loveda_root):
    ds = LoveDA(root=loveda_root, split="test")
    sample = ds[0]
    assert "image" in sample
    assert "mask" not in sample


def test_loveda_len(loveda_root):
    ds = LoveDA(root=loveda_root, split="train", scene=["urban", "rural"])
    assert len(ds) == 2


def test_loveda_single_scene(loveda_root):
    ds = LoveDA(root=loveda_root, split="train", scene=["rural"])
    assert len(ds) == 1


def test_loveda_invalid_split(loveda_root):
    with pytest.raises(AssertionError):
        LoveDA(root=loveda_root, split="bogus")


def test_loveda_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        LoveDA(root=str(tmp_path), split="train", download=False)


def test_loveda_plot(loveda_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = LoveDA(root=loveda_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()
