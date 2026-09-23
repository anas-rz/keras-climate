import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.esd import EmbeddedSeamlessData, ESDQuantizer


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    path = os.path.join(root, "SDC30_EBD_V001_02VMN_2024.tif")
    transform = from_origin(0, 8, 1, 1)
    # Levels (8, 8, 8, 5, 5, 5) factorize into values in [0, 64000)
    data = np.random.randint(0, 64000, (13, 8, 8)).astype("uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=13,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return root


def test_getitem(prepared_root):
    ds = EmbeddedSeamlessData(prepared_root)
    sample = ds[ds.bounds]
    assert "image" in sample
    # (T, H, W, L) where T=13 raster bands, L=6 quantization levels
    assert tuple(sample["image"].shape) == (13, 8, 8, 6)


def test_len(prepared_root):
    ds = EmbeddedSeamlessData(prepared_root)
    assert len(ds) == 1


def test_and_or(prepared_root):
    ds = EmbeddedSeamlessData(prepared_root)
    assert isinstance(ds & ds, IntersectionDataset)
    assert isinstance(ds | ds, UnionDataset)


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EmbeddedSeamlessData(str(tmp_path))


def test_invalid_index(prepared_root):
    ds = EmbeddedSeamlessData(prepared_root)
    with pytest.raises(IndexError, match="not found in dataset with bounds"):
        ds[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]


def test_quantizer_decodes_within_range():
    quantizer = ESDQuantizer()
    indices = np.random.randint(0, 64000, (4, 4, 2)).astype("int32")
    codes = quantizer.quantize(indices)
    codes_np = np.asarray(codes)
    assert codes_np.shape == (2, 4, 4, 6)
    assert codes_np.min() >= -1.0
    assert codes_np.max() <= 1.0


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = EmbeddedSeamlessData(prepared_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
