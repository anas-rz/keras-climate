import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.fire_risk import FireRisk


@pytest.fixture
def prepared_root(tmp_path):
    base = os.path.join(str(tmp_path), FireRisk.directory)
    for split in FireRisk.splits:
        for cls in FireRisk.classes[:2]:
            class_dir = os.path.join(base, split, cls)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(2):
                array = np.random.randint(0, 255, (16, 16, 3), dtype="uint8")
                Image.fromarray(array).save(os.path.join(class_dir, f"{cls}_{i}.png"))
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = FireRisk(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert "label" in sample


def test_len(prepared_root):
    ds = FireRisk(root=prepared_root, split="train", download=False)
    assert len(ds) == 4


def test_invalid_split():
    with pytest.raises(AssertionError):
        FireRisk(split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        FireRisk(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = FireRisk(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
