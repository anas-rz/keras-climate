import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.seco import SeasonalContrastS2


def _write_band_tif(path, size=4, value=10):
    transform = from_origin(0, size, 1, 1)
    data = np.full((1, size, size), value, dtype="uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    base = os.path.join(root, "seasonal_contrast_100k", "000000")
    for subdir in ("0", "1"):
        d = os.path.join(base, subdir)
        os.makedirs(d, exist_ok=True)
        for band in ("B4", "B3", "B2"):
            _write_band_tif(os.path.join(d, f"{band}.tif"))
    return root


def test_getitem_shape_and_dtype(prepared_root):
    ds = SeasonalContrastS2(
        root=prepared_root, version="100k", seasons=1, bands=("B4", "B3", "B2")
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (264, 264, 3)


def test_len():
    ds_100k_len = SeasonalContrastS2.__len__

    class Dummy:
        version = "100k"

    assert ds_100k_len(Dummy()) == 10**5 // 5


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SeasonalContrastS2(root=str(tmp_path), download=False)


def test_invalid_version_raises(prepared_root):
    with pytest.raises(AssertionError):
        SeasonalContrastS2(root=prepared_root, version="bogus")


def test_invalid_band_raises(prepared_root):
    with pytest.raises(AssertionError):
        SeasonalContrastS2(root=prepared_root, bands=("bogus",))


def test_plot_missing_rgb_raises(prepared_root):
    ds = SeasonalContrastS2(root=prepared_root, bands=("B4",))
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SeasonalContrastS2(root=prepared_root, bands=("B4", "B3", "B2"))
    ds.plot(ds[0], suptitle="Test")
    plt.close()


def test_plot_prediction_raises(prepared_root):
    ds = SeasonalContrastS2(root=prepared_root, bands=("B4", "B3", "B2"))
    sample = ds[0]
    sample["prediction"] = sample["image"]
    with pytest.raises(ValueError):
        ds.plot(sample)
