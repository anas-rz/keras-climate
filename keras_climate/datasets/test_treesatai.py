import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.treesatai import TreeSatAI


def _write_tif(path, bands, size=4):
    transform = from_origin(0, size, 1, 1)
    data = np.random.rand(bands, size, size).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    files = ["a.tif", "b.tif"]

    for sensor, bands in [("aerial", 4), ("s1", 3), ("s2", 12)]:
        sensor_dir = os.path.join(root, sensor, "60m")
        os.makedirs(sensor_dir, exist_ok=True)
        for fname in files:
            _write_tif(os.path.join(sensor_dir, fname), bands)

    labels_dir = os.path.join(root, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    labels = {"a.tif": [["Abies", 1.0]], "b.tif": [["Acer", 0.5], ["Betula", 0.5]]}
    with open(
        os.path.join(labels_dir, "TreeSatBA_v9_60m_multi_labels.json"), "w"
    ) as f:
        json.dump(labels, f)

    with open(os.path.join(root, "train_filenames.lst"), "w") as f:
        f.write("\n".join(files))
    with open(os.path.join(root, "test_filenames.lst"), "w") as f:
        f.write("\n".join(files))

    return root


def test_getitem(prepared_root):
    ds = TreeSatAI(root=prepared_root, split="train")
    sample = ds[0]
    assert tuple(sample["image_aerial"].shape) == (4, 4, 4)
    assert tuple(sample["image_s1"].shape) == (4, 4, 3)
    assert tuple(sample["image_s2"].shape) == (4, 4, 12)
    assert tuple(sample["label"].shape) == (len(TreeSatAI.classes),)


def test_len(prepared_root):
    ds = TreeSatAI(root=prepared_root, split="train")
    assert len(ds) == 2


def test_invalid_sensors():
    with pytest.raises(AssertionError):
        TreeSatAI(sensors=("bogus",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        TreeSatAI(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = TreeSatAI(root=prepared_root, split="train")
    ds.plot(ds[0], suptitle="Test")
    plt.close()
