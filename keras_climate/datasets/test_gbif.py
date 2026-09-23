import os

import pandas as pd
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.gbif import GBIF
from keras_climate.datasets.geo import IntersectionDataset, UnionDataset


@pytest.fixture
def prepared_root(tmp_path):
    columns = [
        "gbifID",
        "decimalLatitude",
        "decimalLongitude",
        "day",
        "month",
        "year",
    ]
    rows = [
        [1, 41.881832, -87.623177, "16", "4", "2022"],
        [2, 41.881832, -87.623177, "1", "1", "2022"],
    ]
    df = pd.DataFrame(rows, columns=columns)
    path = os.path.join(str(tmp_path), "0123456-012345678901234.csv")
    df.to_csv(path, sep="\t", index=False)
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = GBIF(prepared_root)
    x = ds[ds.bounds]
    assert isinstance(x, dict)
    assert tuple(x["keypoints"].shape) == (2, 2)


def test_len(prepared_root):
    ds = GBIF(prepared_root)
    assert len(ds) == 2


def test_and(prepared_root):
    ds = GBIF(prepared_root)
    combined = ds & ds
    assert isinstance(combined, IntersectionDataset)


def test_or(prepared_root):
    ds = GBIF(prepared_root)
    combined = ds | ds
    assert isinstance(combined, UnionDataset)


def test_no_data(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        GBIF(str(tmp_path))


def test_invalid_index(prepared_root):
    ds = GBIF(prepared_root)
    with pytest.raises(IndexError, match=r"index: .* not found in dataset with bounds:"):
        ds[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = GBIF(prepared_root)
    sample = ds[ds.bounds]
    fig = ds.plot(sample, suptitle="test")
    assert fig is not None
    plt.close()
