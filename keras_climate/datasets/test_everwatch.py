import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.everwatch import EverWatch


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    base = os.path.join(root, EverWatch.dir)
    os.makedirs(base, exist_ok=True)

    image = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype="uint8"))
    image.save(os.path.join(base, "img1.png"))

    for split in EverWatch.valid_splits:
        df = pd.DataFrame(
            {
                "image_path": ["img1.png", "img1.png"],
                "xmin": [1, 5],
                "ymin": [1, 5],
                "xmax": [10, 15],
                "ymax": [10, 15],
                "label": ["White Ibis", "Great Egret"],
            }
        )
        df.to_csv(os.path.join(base, f"{split}.csv"), index=False)

    return root


def test_getitem(prepared_root):
    ds = EverWatch(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (32, 32, 3)
    assert tuple(sample["bbox_xyxy"].shape) == (2, 4)
    assert tuple(sample["label"].shape) == (2,)


def test_len(prepared_root):
    ds = EverWatch(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_invalid_split():
    with pytest.raises(AssertionError):
        EverWatch(split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EverWatch(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = EverWatch(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
