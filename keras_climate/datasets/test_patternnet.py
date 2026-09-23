import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.patternnet import PatternNet


@pytest.fixture
def prepared_root(tmp_path):
    base = os.path.join(str(tmp_path), PatternNet.directory)
    classes = ("airplane", "beach")
    for cls in classes:
        class_dir = os.path.join(base, cls)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(2):
            path = os.path.join(class_dir, f"{cls}_{i}.jpg")
            data = np.random.randint(0, 255, (16, 16, 3)).astype("uint8")
            Image.fromarray(data).save(path)
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = PatternNet(root=prepared_root, download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert "label" in sample


def test_len(prepared_root):
    ds = PatternNet(root=prepared_root, download=False)
    assert len(ds) == 4


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        PatternNet(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = PatternNet(root=prepared_root, download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
