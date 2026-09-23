import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.etci2021 import ETCI2021

SPLIT_DIRS = {"train": "train", "val": "test", "test": "test_internal"}


def _write_png(path, array, mode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


@pytest.fixture(params=["train", "val", "test"])
def split(request):
    return request.param


@pytest.fixture
def prepared_root(tmp_path, split):
    root = str(tmp_path)
    directory = SPLIT_DIRS[split]
    tile_dir = os.path.join(root, directory, "region1", "tiles")

    rgb = (np.random.rand(4, 4, 3) * 255).astype("uint8")
    mask = np.zeros((4, 4), dtype="uint8")

    _write_png(os.path.join(tile_dir, "vv", "0.png"), rgb, "RGB")
    _write_png(os.path.join(tile_dir, "vh", "0.png"), rgb, "RGB")
    _write_png(os.path.join(tile_dir, "water_body_label", "0.png"), mask, "L")
    if split != "test":
        _write_png(os.path.join(tile_dir, "flood_label", "0.png"), mask, "L")

    return root


def test_getitem(prepared_root, split):
    ds = ETCI2021(root=prepared_root, split=split, download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 4, 6)
    if split != "test":
        assert tuple(sample["mask"].shape) == (4, 4, 2)
    else:
        assert tuple(sample["mask"].shape) == (4, 4, 1)


def test_len(prepared_root, split):
    ds = ETCI2021(root=prepared_root, split=split, download=False)
    assert len(ds) == 1


def test_not_downloaded(tmp_path, split):
    with pytest.raises(DatasetNotFoundError):
        ETCI2021(root=str(tmp_path), split=split, download=False)


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        ETCI2021(root=prepared_root, split="not-a-split", download=False)


def test_plot(prepared_root, split):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ETCI2021(root=prepared_root, split=split, download=False)
    x = ds[0]
    ds.plot(x, suptitle="Test")
    plt.close()
