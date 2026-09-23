import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.vaihingen import Vaihingen2D


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    top_dir = os.path.join(root, Vaihingen2D.image_root)
    os.makedirs(top_dir, exist_ok=True)

    for split, names in Vaihingen2D.splits.items():
        # Only materialize the first file of each split to keep the fixture small
        name = names[0]
        img = np.random.randint(0, 255, (8, 8, 3)).astype("uint8")
        Image.fromarray(img, mode="RGB").save(os.path.join(top_dir, name))

        mask = np.zeros((8, 8, 3), dtype="uint8")
        mask[:4, :, :] = Vaihingen2D.colormap[1]
        mask[4:, :, :] = Vaihingen2D.colormap[2]
        Image.fromarray(mask, mode="RGB").save(os.path.join(root, name))

    return root


def test_getitem(prepared_root):
    ds = Vaihingen2D(root=prepared_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(prepared_root):
    ds = Vaihingen2D(root=prepared_root, split="train")
    assert len(ds) == 1

    ds = Vaihingen2D(root=prepared_root, split="test")
    assert len(ds) == 1


def test_invalid_split():
    with pytest.raises(AssertionError):
        Vaihingen2D(split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        Vaihingen2D(root=str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = Vaihingen2D(root=prepared_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()
