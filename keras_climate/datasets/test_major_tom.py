import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.major_tom import MajorTOMEmbeddings


@pytest.fixture
def embeddings_root(tmp_path):
    size = 2
    embed = 16
    x = np.arange(size)
    y = np.arange(size)
    t = pd.date_range("2018-01-01", periods=size * size)
    embedding = np.random.rand(size * size, embed)
    X, Y = np.meshgrid(x, y)
    xf, yf = X.flatten(), Y.flatten()
    data = {
        "embedding": list(embedding),
        "centre_lon": xf,
        "centre_lat": yf,
        "timestamp": t,
    }
    geometry = gpd.points_from_xy(xf, yf).buffer(0.5).envelope
    gdf = gpd.GeoDataFrame(data, geometry=geometry)

    directory = os.path.join(str(tmp_path), "embeddings")
    os.makedirs(directory, exist_ok=True)
    gdf.to_parquet(os.path.join(directory, "part_00001.parquet"), compression="snappy")
    return directory


def test_major_tom_getitem(embeddings_root):
    ds = MajorTOMEmbeddings(root=embeddings_root)
    sample = ds[0]
    assert "embedding" in sample
    assert "x" in sample
    assert "y" in sample
    assert "t" in sample


def test_major_tom_len(embeddings_root):
    ds = MajorTOMEmbeddings(root=embeddings_root)
    assert len(ds) == 4


def test_major_tom_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        MajorTOMEmbeddings(root=str(tmp_path / "does-not-exist"))


def test_major_tom_plot(embeddings_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = MajorTOMEmbeddings(root=embeddings_root)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
