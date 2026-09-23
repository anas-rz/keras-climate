import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.cowc import COWC, COWCCounting, COWCDetection

SIZE = 8


def test_abstract_not_instantiable():
    with pytest.raises(TypeError):
        COWC()


@pytest.fixture(params=[COWCCounting, COWCDetection])
def dataset_cls(request):
    return request.param


@pytest.fixture(params=["train", "test"])
def prepared_root(tmp_path, dataset_cls, request):
    split = request.param
    root = str(tmp_path)

    filenames = ["car_0.png", "car_1.png"]
    labels = ["1", "0"]
    for fname in filenames:
        data = np.random.randint(0, 255, (SIZE, SIZE, 3)).astype("uint8")
        Image.fromarray(data).save(os.path.join(root, fname))

    list_path = os.path.join(root, dataset_cls.filename.format(split))
    with open(list_path, "w", newline="") as f:
        for fname, label in zip(filenames, labels):
            f.write(f"{fname} {label}\n")

    # _check_integrity() checks for every declared filename with checksum=False
    for fname in dataset_cls.filenames:
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            open(path, "wb").close()

    return root, split


def test_getitem(dataset_cls, prepared_root):
    root, split = prepared_root
    ds = dataset_cls(root=root, split=split, checksum=False)
    x = ds[0]
    assert tuple(x["image"].shape) == (SIZE, SIZE, 3)
    assert "label" in x


def test_len(dataset_cls, prepared_root):
    root, split = prepared_root
    ds = dataset_cls(root=root, split=split, checksum=False)
    assert len(ds) == 2


def test_invalid_split(tmp_path, dataset_cls):
    with pytest.raises(AssertionError):
        dataset_cls(root=str(tmp_path), split="bogus")


def test_out_of_bounds(dataset_cls, prepared_root):
    root, split = prepared_root
    ds = dataset_cls(root=root, split=split, checksum=False)
    with pytest.raises(IndexError):
        ds[2]


def test_not_downloaded(tmp_path, dataset_cls):
    with pytest.raises(DatasetNotFoundError):
        dataset_cls(root=str(tmp_path), download=False, checksum=False)


def test_plot(dataset_cls, prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, split = prepared_root
    ds = dataset_cls(root=root, split=split, checksum=False)
    x = ds[0]
    ds.plot(x, suptitle="Test")
    plt.close()
    ds.plot(x, show_titles=False)
    plt.close()
    x["prediction"] = x["label"]
    ds.plot(x)
    plt.close()
