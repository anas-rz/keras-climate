import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.levircd import LEVIRCD, LEVIRCDPlus


def _create_image(path):
    Z = np.random.randint(255, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(Z).convert("RGB").save(path)


def _create_mask(path):
    Z = np.random.randint(2, size=(16, 16, 3), dtype=np.uint8) * 255
    Image.fromarray(Z).convert("L").save(path)


@pytest.fixture
def levircd_root(tmp_path):
    base = str(tmp_path)
    for directory in ("A", "B", "label"):
        os.makedirs(os.path.join(base, directory))
    for split in ("train", "test"):
        for i in range(2):
            _create_image(os.path.join(base, "A", f"{split}_{i}.png"))
            _create_image(os.path.join(base, "B", f"{split}_{i}.png"))
            _create_mask(os.path.join(base, "label", f"{split}_{i}.png"))
    return base


def test_levircd_getitem(levircd_root):
    ds = LEVIRCD(root=levircd_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, 16, 16, 3)
    assert tuple(sample["mask"].shape) == (16, 16)


def test_levircd_len(levircd_root):
    ds = LEVIRCD(root=levircd_root, split="train")
    assert len(ds) == 2


def test_levircd_invalid_split(levircd_root):
    with pytest.raises(AssertionError):
        LEVIRCD(root=levircd_root, split="bogus")


def test_levircd_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        LEVIRCD(root=str(tmp_path), split="train", download=False)


def test_levircd_plot(levircd_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = LEVIRCD(root=levircd_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()


@pytest.fixture
def levircdplus_root(tmp_path):
    base = str(tmp_path)
    directory = "LEVIR-CD+"
    for split in ("train", "test"):
        for sub in ("A", "B", "label"):
            os.makedirs(os.path.join(base, directory, split, sub))
        for i in range(2):
            fname = f"0{i}.png"
            _create_image(os.path.join(base, directory, split, "A", fname))
            _create_image(os.path.join(base, directory, split, "B", fname))
            _create_mask(os.path.join(base, directory, split, "label", fname))
    return base


def test_levircdplus_getitem(levircdplus_root):
    ds = LEVIRCDPlus(root=levircdplus_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (2, 16, 16, 3)
    assert tuple(sample["mask"].shape) == (16, 16)


def test_levircdplus_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        LEVIRCDPlus(root=str(tmp_path), split="train", download=False)
