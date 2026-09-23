import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.clay import ClayEmbeddings

SIZE = 2


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    x = np.arange(SIZE)
    y = np.arange(SIZE)
    X, Y = np.meshgrid(x, y)
    x = X.flatten()
    y = Y.flatten()
    ids = np.arange(SIZE * SIZE)
    t = pd.date_range("2018-01-01", periods=SIZE * SIZE)

    embed = 8
    embedding = np.random.rand(SIZE * SIZE, embed)
    data = {"id": ids, "date": t, "embeddings": list(embedding)}
    geometry = gpd.points_from_xy(x, y).buffer(0.05).envelope
    gdf = gpd.GeoDataFrame(data, geometry=geometry)
    gdf.to_parquet(os.path.join(root, "data.parquet"))

    return root


def test_getitem(prepared_root):
    ds = ClayEmbeddings(root=prepared_root)
    sample = ds[0]
    assert tuple(sample["embedding"].shape) == (8,)
    assert "x" in sample
    assert "y" in sample
    assert "t" in sample


def test_len(prepared_root):
    ds = ClayEmbeddings(root=prepared_root)
    assert len(ds) == SIZE * SIZE


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        ClayEmbeddings(root=os.path.join(str(tmp_path), "does_not_exist"))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ClayEmbeddings(root=prepared_root)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
