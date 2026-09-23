import json
import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.satlas import SatlasPretrain


@pytest.fixture
def prepared_root(tmp_path, monkeypatch):
    # Keep chips tiny for a fast test.
    monkeypatch.setattr(SatlasPretrain, "chip_size", 8)

    root = str(tmp_path)
    col, row = 100, 200
    directory = "ts1"

    for band in SatlasPretrain.bands["sentinel1"]:
        band_dir = os.path.join(root, "sentinel1", directory, band)
        os.makedirs(band_dir, exist_ok=True)
        img = Image.fromarray(np.zeros((4, 4), dtype="uint8"), mode="L")
        img.save(os.path.join(band_dir, f"{col}_{row}.png"))

    static_dir = os.path.join(root, "static", f"{col}_{row}")
    os.makedirs(static_dir, exist_ok=True)
    label_img = Image.fromarray(np.zeros((8, 8), dtype="uint8"), mode="L")
    label_img.save(os.path.join(static_dir, "land_cover.png"))

    metadata_dir = os.path.join(root, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, "train_lowres.json"), "w") as f:
        json.dump([[col, row]], f)
    with open(os.path.join(metadata_dir, "good_images_lowres_all.json"), "w") as f:
        json.dump([[col, row, directory]], f)
    with open(os.path.join(metadata_dir, "image_times.json"), "w") as f:
        json.dump({directory: "2022-01-01T00:00:00+00:00"}, f)

    return root


def test_getitem(prepared_root):
    ds = SatlasPretrain(
        root=prepared_root,
        images=("sentinel1",),
        labels=("land_cover",),
        download=False,
    )
    sample = ds[0]
    assert tuple(sample["image_sentinel1"].shape) == (8, 8, 2)
    assert tuple(sample["mask_land_cover"].shape) == (8, 8)
    assert "time_sentinel1" in sample


def test_len(prepared_root):
    ds = SatlasPretrain(
        root=prepared_root,
        images=("sentinel1",),
        labels=("land_cover",),
        download=False,
    )
    assert len(ds) == 1


def test_invalid_images(prepared_root):
    with pytest.raises(AssertionError):
        SatlasPretrain(root=prepared_root, images=("foo",), download=False)


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SatlasPretrain(
            root=str(tmp_path),
            images=("sentinel1",),
            labels=(),
            download=False,
        )


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SatlasPretrain(
        root=prepared_root,
        images=("sentinel1",),
        labels=("land_cover",),
        download=False,
    )
    ds.plot(ds[0], suptitle="Test")
    plt.close()
