import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.ucmerced import UCMerced


@pytest.fixture
def prepared_root(tmp_path):
    base = os.path.join(str(tmp_path), UCMerced.base_dir)
    os.makedirs(base, exist_ok=True)
    classes = ("agricultural", "forest")
    all_files = []
    for cls in classes:
        class_dir = os.path.join(base, cls)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(2):
            fname = f"{cls}{i}.tif"
            path = os.path.join(class_dir, fname)
            data = np.random.randint(0, 255, (16, 16, 3)).astype("uint8")
            Image.fromarray(data, mode="RGB").save(path)
            all_files.append(fname)

    half = len(all_files) // 2
    splits = {"train": all_files[:half], "val": all_files[:half], "test": all_files[half:]}
    for split, fnames in splits.items():
        with open(
            os.path.join(str(tmp_path), UCMerced.split_filenames[split]), "w"
        ) as f:
            f.write("\n".join(fnames) + "\n")

    return str(tmp_path)


def test_getitem(prepared_root):
    ds = UCMerced(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (256, 256, 3)
    assert "label" in sample


def test_len(prepared_root):
    ds = UCMerced(root=prepared_root, split="train", download=False)
    assert len(ds) == 2


def test_invalid_split():
    with pytest.raises(AssertionError):
        UCMerced(split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        UCMerced(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = UCMerced(root=prepared_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()
