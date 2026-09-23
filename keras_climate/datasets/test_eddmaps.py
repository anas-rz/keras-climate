import os

import pandas as pd
import pytest

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.eddmaps import EDDMapS


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    df = pd.DataFrame(
        {
            "ObsDate": ["01-01-21", "02-15-21"],
            "Latitude": [40.0, 41.0],
            "Longitude": [-79.0, -80.0],
        }
    )
    df.to_csv(os.path.join(root, "mappings.csv"), index=False)
    return root


def test_getitem(prepared_root):
    ds = EDDMapS(root=prepared_root)
    sample = ds[ds.bounds]
    assert isinstance(sample, dict)
    assert tuple(sample["keypoints"].shape) == (2, 2)


def test_len(prepared_root):
    ds = EDDMapS(root=prepared_root)
    assert len(ds) == 2


def test_and_or(prepared_root):
    ds = EDDMapS(root=prepared_root)
    assert isinstance(ds & ds, IntersectionDataset)
    assert isinstance(ds | ds, UnionDataset)


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EDDMapS(root=str(tmp_path))


def test_invalid_index(prepared_root):
    ds = EDDMapS(root=prepared_root)
    with pytest.raises(IndexError, match="not found in dataset with bounds"):
        ds[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = EDDMapS(root=prepared_root)
    ds.plot(ds[ds.bounds], suptitle="Test")
    plt.close()
