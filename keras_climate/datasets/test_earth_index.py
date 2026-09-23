import os

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.earth_index import EarthIndexEmbeddings


@pytest.fixture
def parquet_path(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "embedding": [np.random.rand(8).astype("float32") for _ in range(3)],
            "geometry": [Point(-79.0, 40.0), Point(-80.0, 41.0), Point(-81.0, 42.0)],
        },
        crs="EPSG:4326",
    )
    path = os.path.join(str(tmp_path), "embeddings.parquet")
    gdf.to_parquet(path)
    return path


def test_getitem(parquet_path):
    ds = EarthIndexEmbeddings(root=parquet_path)
    sample = ds[0]
    assert "embedding" in sample
    assert tuple(sample["embedding"].shape) == (8,)
    assert "x" in sample and "y" in sample


def test_len(parquet_path):
    ds = EarthIndexEmbeddings(root=parquet_path)
    assert len(ds) == 3


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EarthIndexEmbeddings(root=os.path.join(str(tmp_path), "does_not_exist.parquet"))


def test_plot(parquet_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = EarthIndexEmbeddings(root=parquet_path)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
