import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.gid15 import GID15


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    for split in ("train", "val", "test"):
        img_dir = os.path.join(root, "GID", "img_dir", split)
        os.makedirs(img_dir, exist_ok=True)
        ann_dir = os.path.join(root, "GID", "ann_dir", split)
        os.makedirs(ann_dir, exist_ok=True)
        for i in range(2):
            image = Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype="uint8"))
            image.save(os.path.join(img_dir, f"img_{i}.tif"))
            if split != "test":
                mask = Image.fromarray(np.random.randint(0, 15, (16, 16), dtype="uint8"))
                mask.save(os.path.join(ann_dir, f"img_{i}_15label.png"))
    return root


def test_getitem_train(prepared_root):
    ds = GID15(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert tuple(sample["mask"].shape) == (16, 16)


def test_getitem_test(prepared_root):
    ds = GID15(root=prepared_root, split="test", download=False)
    sample = ds[0]
    assert "image" in sample
    assert "mask" not in sample


def test_len(prepared_root):
    ds = GID15(root=prepared_root, split="train", download=False)
    assert len(ds) == 2


def test_invalid_split():
    with pytest.raises(AssertionError):
        GID15(split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        GID15(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = GID15(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
