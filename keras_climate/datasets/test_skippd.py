import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.skippd import SKIPPD


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    for task, image_shape, label_shape in [
        ("nowcast", (64, 64, 3), ()),
        ("forecast", (16, 64, 64, 3), (16,)),
    ]:
        path = os.path.join(root, SKIPPD.data_file_name.format(task))
        with h5py.File(path, "w") as f:
            for split in ("trainval", "test"):
                grp = f.create_group(split)
                n = 3
                grp.create_dataset(
                    "images_log", data=np.random.randint(0, 255, (n, *image_shape))
                )
                grp.create_dataset(
                    "pv_log", data=np.random.rand(n, *label_shape).astype("float32")
                )
    return root


def test_nowcast_getitem(prepared_root):
    ds = SKIPPD(root=prepared_root, split="trainval", task="nowcast", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (64, 64, 3)
    assert "label" in sample


def test_forecast_getitem(prepared_root):
    ds = SKIPPD(root=prepared_root, split="trainval", task="forecast", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (64, 64, 48)


def test_len(prepared_root):
    ds = SKIPPD(root=prepared_root, split="test", task="nowcast", download=False)
    assert len(ds) == 3


def test_invalid_split_raises():
    with pytest.raises(AssertionError):
        SKIPPD(split="bogus")


def test_invalid_task_raises():
    with pytest.raises(AssertionError):
        SKIPPD(task="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SKIPPD(root=str(tmp_path), download=False)


def test_plot_nowcast(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SKIPPD(root=prepared_root, split="trainval", task="nowcast", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()


def test_plot_forecast(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SKIPPD(root=prepared_root, split="trainval", task="forecast", download=False)
    ds.plot(ds[0])
    plt.close()
