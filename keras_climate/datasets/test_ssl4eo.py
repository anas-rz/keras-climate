import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.ssl4eo import SSL4EOL, SSL4EOS12


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
def ssl4eol_root(tmp_path):
    root = str(tmp_path)
    split = "tm_toa"
    subdir = os.path.join(root, f"ssl4eo_l_{split}")
    scene_dir = os.path.join(subdir, "0000000")
    for date in ("20210101", "20210401"):
        season_dir = os.path.join(scene_dir, f"LT05_L1TP_20210101_{date}")
        os.makedirs(season_dir, exist_ok=True)
        _write_tif(os.path.join(season_dir, "all_bands.tif"), 7)
    return root


@pytest.fixture
def ssl4eos12_root(tmp_path):
    root = str(tmp_path)
    split = "s2c"
    scene_dir = os.path.join(root, split, "0000000")
    for date_dir in ("20210101T000000", "20210401T000000"):
        season_dir = os.path.join(scene_dir, date_dir)
        os.makedirs(season_dir, exist_ok=True)
        for band in SSL4EOS12.metadata[split]["bands"]:
            _write_tif(os.path.join(season_dir, f"{band}.tif"), 1)
    return root


def test_ssl4eol_getitem(ssl4eol_root):
    ds = SSL4EOL(root=ssl4eol_root, split="tm_toa", seasons=1)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 4, 7)
    assert tuple(sample["wavelength"].shape) == (7,)


def test_ssl4eol_len(ssl4eol_root):
    ds = SSL4EOL(root=ssl4eol_root, split="tm_toa", seasons=1)
    assert len(ds) == 1


def test_ssl4eol_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SSL4EOL(root=str(tmp_path), split="tm_toa", download=False)


def test_ssl4eol_plot(ssl4eol_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SSL4EOL(root=ssl4eol_root, split="tm_toa", seasons=1)
    ds.plot(ds[0], suptitle="Test")
    plt.close()


def test_ssl4eos12_getitem(ssl4eos12_root):
    ds = SSL4EOS12(root=ssl4eos12_root, split="s2c", seasons=1)
    sample = ds[0]
    num_bands = len(SSL4EOS12.metadata["s2c"]["bands"])
    assert tuple(sample["image"].shape) == (SSL4EOS12.size, SSL4EOS12.size, num_bands)


def test_ssl4eos12_len(ssl4eos12_root):
    ds = SSL4EOS12(root=ssl4eos12_root, split="s2c", seasons=1)
    assert len(ds) == 251079


def test_ssl4eos12_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SSL4EOS12(root=str(tmp_path), split="s2c", download=False)


def test_ssl4eos12_plot(ssl4eos12_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SSL4EOS12(root=ssl4eos12_root, split="s2c", seasons=1)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
