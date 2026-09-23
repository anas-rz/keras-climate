import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.seasonet import SeasoNet


def _write_tif(path, size=4, count=3, dtype="uint16", value=10):
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
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    rows = []
    for i, rel in enumerate(["p0", "p1"]):
        basename = rel
        loc_dir = os.path.join(root, rel)
        os.makedirs(loc_dir, exist_ok=True)
        base_path = os.path.join(loc_dir, basename)
        for band, count in SeasoNet.band_nums.items():
            _write_tif(f"{base_path}_{band}.tif", size=4, count=count)
        _write_tif(f"{base_path}_labels.tif", size=4, count=1, value=2)
        rows.append(
            {
                "Index": i,
                "Path": rel,
                "Grid": 1,
                "Season": "Spring",
                "Latitude": i,
                "Longitude": i,
            }
        )
    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(root, "meta.csv"), index=False)

    splits_dir = os.path.join(root, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    for season_dir in ("spring", "summer", "fall", "winter", "snow"):
        os.makedirs(os.path.join(root, season_dir), exist_ok=True)
    pd.Series([0, 1]).to_csv(
        os.path.join(splits_dir, "train.csv"), header=False, index=False
    )
    return root


def test_getitem_shape_and_dtype(prepared_root):
    ds = SeasoNet(root=prepared_root, split="train", bands=("10m_RGB",), download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (1, 120, 120, 3)
    assert tuple(sample["mask"].shape) == (4, 4)


def test_len(prepared_root):
    ds = SeasoNet(root=prepared_root, split="train", bands=("10m_RGB",), download=False)
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SeasoNet(root=str(tmp_path), download=False)


def test_plot_missing_rgb_raises(prepared_root):
    ds = SeasoNet(root=prepared_root, split="train", bands=("20m",), download=False)
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SeasoNet(root=prepared_root, split="train", bands=("10m_RGB",), download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
