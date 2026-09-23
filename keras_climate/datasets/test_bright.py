import os

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.bright import BRIGHTDFC2025

SIZE = 8
TRANSFORM = Affine(4.57e-06, 0.0, 9.79, 0.0, -4.57e-06, 1.84)


def _write_tif(path, channels):
    data = np.random.randint(0, 4, (channels, SIZE, SIZE), dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=channels,
        crs=CRS.from_epsg(4326),
        dtype=data.dtype,
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)


def _populate(root, data_dir, ids, dir_name, with_target):
    for sid in ids:
        pre = os.path.join(root, data_dir, dir_name, "pre-event", f"{sid}_pre_disaster.tif")
        _write_tif(pre, 3)
        post = os.path.join(
            root, data_dir, dir_name, "post-event", f"{sid}_post_disaster.tif"
        )
        _write_tif(post, 1)
        if with_target:
            target = os.path.join(
                root, data_dir, dir_name, "target", f"{sid}_building_damage.tif"
            )
            _write_tif(target, 1)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    data_dir = BRIGHTDFC2025.data_dir

    for sub in ("train/pre-event", "train/post-event", "train/target",
                "val/pre-event", "val/post-event"):
        os.makedirs(os.path.join(root, data_dir, sub), exist_ok=True)

    train_ids = ["train_0000", "train_0001"]
    val_ids = ["val_0000"]

    _populate(root, data_dir, train_ids, "train", with_target=True)
    _populate(root, data_dir, val_ids, "val", with_target=False)

    with open(os.path.join(root, data_dir, "train_setlevel.txt"), "w") as f:
        f.writelines(f"{sid}\n" for sid in train_ids)
    with open(os.path.join(root, data_dir, "holdout_setlevel.txt"), "w") as f:
        f.write("")
    with open(os.path.join(root, data_dir, "val_setlevel.txt"), "w") as f:
        f.writelines(f"{sid}\n" for sid in val_ids)

    return root


def test_getitem(prepared_root):
    ds = BRIGHTDFC2025(prepared_root, split="train")
    x = ds[0]
    assert tuple(x["image"].shape) == (2, SIZE, SIZE, 3)
    assert tuple(x["mask"].shape) == (SIZE, SIZE)


def test_getitem_test_split(prepared_root):
    ds = BRIGHTDFC2025(prepared_root, split="test")
    x = ds[0]
    assert "mask" not in x
    assert tuple(x["image"].shape) == (2, SIZE, SIZE, 3)


def test_len(prepared_root):
    assert len(BRIGHTDFC2025(prepared_root, split="train")) == 2
    assert len(BRIGHTDFC2025(prepared_root, split="val")) == 0
    assert len(BRIGHTDFC2025(prepared_root, split="test")) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        BRIGHTDFC2025(str(tmp_path))


def test_invalid_split():
    with pytest.raises(AssertionError):
        BRIGHTDFC2025(split="bogus")


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = BRIGHTDFC2025(prepared_root, split="train")
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()

    sample["prediction"] = sample["mask"]
    ds.plot(sample)
    plt.close()
