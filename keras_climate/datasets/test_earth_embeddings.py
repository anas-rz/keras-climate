import os

import numpy as np
import pandas as pd
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.earth_embeddings import EarthEmbeddings


@pytest.fixture
def parquet_path(tmp_path):
    df = pd.DataFrame(
        {
            "embedding": [np.random.rand(8).astype("float32") for _ in range(3)],
            "centre_lon": [-79.0, -80.0, -81.0],
            "centre_lat": [40.0, 41.0, 42.0],
            "timestamp": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
        }
    )
    path = os.path.join(str(tmp_path), "embeddings.parquet")
    df.to_parquet(path)
    return path


def test_getitem(parquet_path):
    ds = EarthEmbeddings(root=parquet_path)
    sample = ds[0]
    assert "embedding" in sample
    assert tuple(sample["embedding"].shape) == (8,)
    assert "x" in sample and "y" in sample and "t" in sample


def test_len(parquet_path):
    ds = EarthEmbeddings(root=parquet_path)
    assert len(ds) == 3


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EarthEmbeddings(root=os.path.join(str(tmp_path), "does_not_exist.parquet"))


def test_plot(parquet_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = EarthEmbeddings(root=parquet_path)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
