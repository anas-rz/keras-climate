import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.cv4a_kenya_crop_type import CV4AKenyaCropType


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(CV4AKenyaCropType, "tiles", ("0",))
    monkeypatch.setattr(CV4AKenyaCropType, "dates", ("20190606",))
    monkeypatch.setattr(CV4AKenyaCropType, "tile_height", 2)
    monkeypatch.setattr(CV4AKenyaCropType, "tile_width", 2)

    root = str(tmp_path)
    with open(os.path.join(root, "FieldIds.csv"), "w") as f:
        f.write("field_id\n1\n")

    tile_dir = os.path.join(root, "data", "0")
    os.makedirs(tile_dir, exist_ok=True)
    Image.fromarray(np.random.randint(0, 255, (2, 2), dtype="int32")).save(
        os.path.join(tile_dir, "0_field_id.tif")
    )
    Image.fromarray(np.random.randint(0, 255, (2, 2), dtype="uint8")).save(
        os.path.join(tile_dir, "0_label.tif")
    )

    date_dir = os.path.join(tile_dir, "20190606")
    os.makedirs(date_dir, exist_ok=True)
    for band in CV4AKenyaCropType.all_bands:
        arr = (np.random.rand(2, 2) * 1000).astype("float32")
        Image.fromarray(arr).save(os.path.join(date_dir, f"0_{band}_20190606.tif"))

    return CV4AKenyaCropType(root=root, download=False)


def test_getitem(dataset):
    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample
    assert tuple(sample["image"].shape) == (1, 2, 2, len(CV4AKenyaCropType.all_bands))


def test_len(dataset):
    assert len(dataset) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CV4AKenyaCropType(root=str(tmp_path), download=False)


def test_invalid_bands():
    with pytest.raises(AssertionError):
        CV4AKenyaCropType(bands=("foo", "bar"))


def test_plot(dataset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = dataset[0]
    dataset.plot(sample, time_step=0, suptitle="Test")
    plt.close()


def test_plot_rgb_missing(dataset):
    dataset = CV4AKenyaCropType(root=dataset.root, bands=("B01",))
    with pytest.raises(RGBBandsMissingError):
        dataset.plot(dataset[0])
