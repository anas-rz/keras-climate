import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.geonrw import GeoNRW


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    # override a small city list so fixtures stay tiny
    city = GeoNRW.train_list[0]
    city_dir = os.path.join(root, city)
    os.makedirs(city_dir, exist_ok=True)

    rgb = Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype="uint8"))
    rgb.save(os.path.join(city_dir, "1_1_rgb.jp2"))

    dem = Image.fromarray(np.random.rand(16, 16).astype("float32"), mode="F")
    dem.save(os.path.join(city_dir, "1_1_dem.tif"))

    seg = Image.fromarray(np.random.randint(0, 10, (16, 16)).astype("int32"), mode="I")
    seg.save(os.path.join(city_dir, "1_1_seg.tif"))

    return root


def test_getitem(monkeypatch, prepared_root):
    monkeypatch.setattr(GeoNRW, "train_list", (GeoNRW.train_list[0],))
    ds = GeoNRW(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert tuple(sample["mask"].shape) == (16, 16)
    assert tuple(sample["dem"].shape) == (16, 16, 1)


def test_len(monkeypatch, prepared_root):
    monkeypatch.setattr(GeoNRW, "train_list", (GeoNRW.train_list[0],))
    ds = GeoNRW(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_invalid_split():
    with pytest.raises(AssertionError):
        GeoNRW(split="bogus")


def test_not_downloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(GeoNRW, "train_list", (GeoNRW.train_list[0],))
    with pytest.raises(DatasetNotFoundError):
        GeoNRW(root=str(tmp_path), download=False)


def test_plot(monkeypatch, prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(GeoNRW, "train_list", (GeoNRW.train_list[0],))
    ds = GeoNRW(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
