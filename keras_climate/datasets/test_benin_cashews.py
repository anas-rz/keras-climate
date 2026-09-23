import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.benin_cashews import BeninSmallHolderCashews


@pytest.fixture
def prepared_root(tmp_path, monkeypatch):
    monkeypatch.setattr(BeninSmallHolderCashews, "dates", ("20191105",))
    monkeypatch.setattr(BeninSmallHolderCashews, "tile_height", 4)
    monkeypatch.setattr(BeninSmallHolderCashews, "tile_width", 4)

    root = str(tmp_path)
    imagery_dir = os.path.join(root, "imagery", "00", "00_20191105")
    os.makedirs(imagery_dir, exist_ok=True)

    transform = from_origin(0, 4, 1, 1)
    for band in BeninSmallHolderCashews.all_bands:
        path = os.path.join(imagery_dir, f"00_20191105_{band}_10m.tif")
        data = np.random.randint(0, 255, (1, 4, 4)).astype("uint16")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

    labels_dir = os.path.join(root, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"class": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
                },
            }
        ],
    }
    with open(os.path.join(labels_dir, "00.geojson"), "w") as f:
        json.dump(geojson, f)

    return root


def test_getitem(prepared_root):
    ds = BeninSmallHolderCashews(prepared_root, chip_size=4, stride=4)
    x = ds[0]
    assert x["image"].shape[-1] == len(BeninSmallHolderCashews.all_bands)
    assert x["image"].shape[0] == 1  # single date
    assert tuple(x["mask"].shape) == (4, 4)
    assert "x" in x
    assert "y" in x


def test_len(prepared_root):
    ds = BeninSmallHolderCashews(prepared_root, chip_size=4, stride=4)
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        BeninSmallHolderCashews(str(tmp_path))


def test_invalid_bands():
    with pytest.raises(AssertionError):
        BeninSmallHolderCashews(bands=("foo", "bar"))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = BeninSmallHolderCashews(prepared_root, chip_size=4, stride=4)
    x = ds[0]
    ds.plot(x, suptitle="Test")
    plt.close()


def test_failed_plot(prepared_root):
    ds = BeninSmallHolderCashews(
        prepared_root, chip_size=4, stride=4, bands=("B01",)
    )
    x = ds[0]
    with pytest.raises(RGBBandsMissingError):
        ds.plot(x)
