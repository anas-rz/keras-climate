import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.deepglobelandcover import DeepGlobeLandCover


@pytest.fixture(params=["train", "test"])
def split(request):
    return request.param


@pytest.fixture
def prepared_root(tmp_path, split):
    root = str(tmp_path)
    split_folder = "training_data" if split == "train" else "test_data"
    images_dir = os.path.join(root, "data", split_folder, "images")
    masks_dir = os.path.join(root, "data", split_folder, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for i in range(2):
        img = np.random.randint(0, 255, (8, 8, 3), dtype="uint8")
        Image.fromarray(img).save(os.path.join(images_dir, f"{i}_sat.jpg"))

        mask = np.zeros((8, 8, 3), dtype="uint8")
        color = DeepGlobeLandCover.colormap[i % len(DeepGlobeLandCover.colormap)]
        mask[:, :] = color
        Image.fromarray(mask).save(os.path.join(masks_dir, f"{i}_mask.png"))

    return root


def test_getitem(prepared_root, split):
    ds = DeepGlobeLandCover(root=prepared_root, split=split)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(prepared_root, split):
    ds = DeepGlobeLandCover(root=prepared_root, split=split)
    assert len(ds) == 2


def test_invalid_split():
    with pytest.raises(AssertionError):
        DeepGlobeLandCover(split="bad")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        DeepGlobeLandCover(root=str(tmp_path))


def test_plot(prepared_root, split):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DeepGlobeLandCover(root=prepared_root, split=split)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()

    sample["prediction"] = sample["mask"]
    ds.plot(sample)
    plt.close()
