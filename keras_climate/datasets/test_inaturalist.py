import os

import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.inaturalist import INaturalist


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    csv_path = os.path.join(root, "observations.csv")
    with open(csv_path, "w") as f:
        f.write("observed_on,time_observed_at,latitude,longitude\n")
        f.write("2021-01-01,2021-01-01 10:00:00 +0000,10.0,20.0\n")
        f.write("2021-02-02,2021-02-02 11:00:00 +0000,11.0,21.0\n")
        # row with missing lat/lon should be dropped
        f.write("2021-03-03,,,30.0\n")
    return root


def test_getitem(prepared_root):
    ds = INaturalist(root=prepared_root)
    sample = ds[ds.bounds]
    assert "keypoints" in sample
    assert tuple(sample["keypoints"].shape) == (2, 2)


def test_len(prepared_root):
    ds = INaturalist(root=prepared_root)
    assert len(ds) == 2


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        INaturalist(root=str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = INaturalist(root=prepared_root)
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()
