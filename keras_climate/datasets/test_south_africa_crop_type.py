import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.south_africa_crop_type import SouthAfricaCropType


def _write_tif(path, size=4, count=1, dtype="uint16", value=10):
    transform = from_origin(0, size, 1, 1)
    data = np.full((count, size, size), value, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=count,
        dtype=dtype,
        crs="EPSG:32734",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    field_id = "1"
    date = "2017_07_01"
    imagery_dir = os.path.join(root, "train", "imagery", "s2", field_id, date)
    os.makedirs(imagery_dir, exist_ok=True)
    for band in ("B04", "B03", "B02"):
        _write_tif(
            os.path.join(imagery_dir, f"{field_id}_{date}_{band}_10m.tif")
        )

    labels_dir = os.path.join(root, "train", "labels")
    os.makedirs(labels_dir, exist_ok=True)
    _write_tif(
        os.path.join(labels_dir, f"{field_id}.tif"), dtype="uint8", value=2
    )
    return root


def test_getitem_shape_and_dtype(prepared_root):
    ds = SouthAfricaCropType(
        paths=prepared_root, bands=("B04", "B03", "B02"), download=False
    )
    x, y, t = ds.bounds
    sample = ds[x, y, t]
    assert tuple(sample["image"].shape) == (4, 4, 3)
    assert tuple(sample["mask"].shape) == (4, 4)
    assert "bounds" in sample and "transform" in sample


def test_len(prepared_root):
    ds = SouthAfricaCropType(
        paths=prepared_root, bands=("B04", "B03", "B02"), download=False
    )
    assert len(ds) == 1


def test_classes_must_include_background(prepared_root):
    with pytest.raises(AssertionError):
        SouthAfricaCropType(
            paths=prepared_root, bands=("B04", "B03", "B02"), classes=[1, 2]
        )


def test_invalid_classes_raise(prepared_root):
    with pytest.raises(AssertionError):
        SouthAfricaCropType(
            paths=prepared_root, bands=("B04", "B03", "B02"), classes=[0, 99]
        )


def test_not_found_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetNotFoundError):
        SouthAfricaCropType(
            paths=str(empty), bands=("B04", "B03", "B02"), download=False
        )


def test_plot_missing_rgb_raises(prepared_root):
    ds = SouthAfricaCropType(paths=prepared_root, bands=("B04",), download=False)
    x, y, t = ds.bounds
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[x, y, t])


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SouthAfricaCropType(
        paths=prepared_root, bands=("B04", "B03", "B02"), download=False
    )
    x, y, t = ds.bounds
    ds.plot(ds[x, y, t], suptitle="Test")
    plt.close()
