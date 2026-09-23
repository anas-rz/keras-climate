import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.sen12ms import SEN12MS


def _write_tif(path, size=4, count=3, value=10):
    transform = from_origin(0, size, 1, 1)
    data = np.full((count, size, size), value, dtype="uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=count,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    scene = "ROIs1158_spring"
    band_counts = {"s1": 2, "s2": 13, "lc": 4}
    for source, count in band_counts.items():
        subdir = os.path.join(root, scene, f"{source}_1")
        os.makedirs(subdir, exist_ok=True)
        _write_tif(
            os.path.join(subdir, f"{scene}_{source}_1_p100.tif"), count=count
        )

    for empty_scene in ("ROIs1868_summer", "ROIs1970_fall", "ROIs2017_winter"):
        os.makedirs(os.path.join(root, empty_scene), exist_ok=True)

    with open(os.path.join(root, "train_list.txt"), "w") as f:
        f.write(f"{scene}_s2_1_p100.tif\n")
    with open(os.path.join(root, "test_list.txt"), "w") as f:
        f.write(f"{scene}_s2_1_p100.tif\n")

    return root


def test_getitem_shape_and_dtype(prepared_root):
    ds = SEN12MS(root=prepared_root, split="train", checksum=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 4, 15)
    assert tuple(sample["mask"].shape) == (4, 4)


def test_len(prepared_root):
    ds = SEN12MS(root=prepared_root, split="train", checksum=False)
    assert len(ds) == 1


def test_band_subset(prepared_root):
    ds = SEN12MS(
        root=prepared_root,
        split="train",
        bands=SEN12MS.BAND_SETS["s2-reduced"],
        checksum=False,
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 4, 6)


def test_invalid_bands_raises():
    with pytest.raises(ValueError):
        SEN12MS(bands=("bogus",))


def test_invalid_split_raises():
    with pytest.raises(AssertionError):
        SEN12MS(split="val")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SEN12MS(root=str(tmp_path), checksum=False)


def test_plot_missing_rgb_raises(prepared_root):
    ds = SEN12MS(
        root=prepared_root, split="train", bands=SEN12MS.BAND_SETS["s1"], checksum=False
    )
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SEN12MS(root=prepared_root, split="train", checksum=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
