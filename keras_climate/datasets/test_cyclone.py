import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.cyclone import TropicalCyclone


@pytest.fixture(params=["train", "test"])
def split(request):
    return request.param


@pytest.fixture
def prepared_root(tmp_path, split):
    root = str(tmp_path)
    split_dir = os.path.join(root, split)
    os.makedirs(split_dir, exist_ok=True)

    filename = "training_set" if split == "train" else "test_set"

    image_ids = [f"img_{i}" for i in range(3)]
    features = pd.DataFrame(
        {
            "image_id": image_ids,
            "storm_id": ["s"] * 3,
            "relative_time": [0, 1, 2],
            "ocean": [1, 2, 1],
        }
    )
    labels = pd.DataFrame({"image_id": image_ids, "wind_speed": [10, 20, 30]})
    features.to_csv(os.path.join(root, f"{filename}_features.csv"), index=False)
    labels.to_csv(os.path.join(root, f"{filename}_labels.csv"), index=False)

    for image_id in image_ids:
        arr = np.random.randint(0, 255, (8, 8), dtype="uint8")
        Image.fromarray(arr).save(os.path.join(split_dir, f"{image_id}.jpg"))

    return root


def test_getitem(prepared_root, split):
    ds = TropicalCyclone(root=prepared_root, split=split, download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (366, 366, 3)
    assert "label" in sample
    assert "relative_time" in sample
    assert "ocean" in sample


def test_len(prepared_root, split):
    ds = TropicalCyclone(root=prepared_root, split=split, download=False)
    assert len(ds) == 3


def test_invalid_split():
    with pytest.raises(AssertionError):
        TropicalCyclone(split="bad")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        TropicalCyclone(root=str(tmp_path), download=False)


def test_plot(prepared_root, split):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = TropicalCyclone(root=prepared_root, split=split, download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
