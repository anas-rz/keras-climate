import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.dota import DOTA


def _write_image(path, size=16):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.randint(0, 255, (size, size, 3), dtype="uint8")
    Image.fromarray(arr).save(path)


def _write_annotation(path, no_boxes=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("imagesource:dummy\n")
        f.write("gsd:1.0\n")
        if not no_boxes:
            f.write("1.0 1.0 5.0 1.0 5.0 5.0 1.0 5.0 plane 0\n")
            f.write("2.0 2.0 6.0 2.0 6.0 6.0 2.0 6.0 ship 0\n")


@pytest.fixture(params=["train", "val"])
def split(request):
    return request.param


@pytest.fixture
def prepared_root(tmp_path, split):
    root = str(tmp_path)
    version = "2.0"

    rows = []
    for i in range(2):
        img_name = f"P{i:04d}.png"
        ann_name = f"P{i:04d}.txt"
        img_path = os.path.join(split, "images", img_name)
        ann_path = os.path.join(split, "annotations", f"version{version}", ann_name)
        _write_image(os.path.join(root, img_path))
        _write_annotation(os.path.join(root, ann_path), no_boxes=(i == 0))
        rows.append({"image_path": img_path, "annotation_path": ann_path, "split": split, "version": version})

    pd.DataFrame(rows).to_csv(os.path.join(root, "samples.csv"), index=False)
    return root


def test_getitem_oriented(prepared_root, split):
    ds = DOTA(root=prepared_root, split=split, version="2.0", bbox_orientation="oriented", download=False)
    sample = ds[1]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert tuple(sample["bbox"].shape) == (2, 8)
    assert tuple(sample["labels"].shape) == (2,)


def test_getitem_horizontal_no_boxes(prepared_root, split):
    ds = DOTA(root=prepared_root, split=split, version="2.0", bbox_orientation="horizontal", download=False)
    sample = ds[0]
    assert tuple(sample["bbox_xyxy"].shape) == (0, 4)
    assert tuple(sample["labels"].shape) == (0,)


def test_len(prepared_root, split):
    ds = DOTA(root=prepared_root, split=split, download=False)
    assert len(ds) == 2


def test_invalid_split():
    with pytest.raises(AssertionError):
        DOTA(split="bad")


def test_invalid_version():
    with pytest.raises(AssertionError):
        DOTA(version="bad")


def test_invalid_orientation():
    with pytest.raises(AssertionError):
        DOTA(bbox_orientation="bad")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        DOTA(root=str(tmp_path), download=False)


def test_plot(prepared_root, split):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DOTA(root=prepared_root, split=split, download=False)
    ds.plot(ds[1], suptitle="Test")
    plt.close()
