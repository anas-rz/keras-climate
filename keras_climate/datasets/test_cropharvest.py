import json
import os

import h5py
import numpy as np
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.cropharvest import CropHarvest


def _make_geojson():
    return {
        "type": "FeatureCollection",
        "crs": {},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "dataset": "TestDataset1",
                    "index": 0,
                    "is_crop": 1,
                    "label": "soybean",
                },
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
            },
            {
                "type": "Feature",
                "properties": {
                    "dataset": "TestDataset1",
                    "index": 1,
                    "is_crop": 1,
                    "label": None,
                },
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
            },
            {
                "type": "Feature",
                "properties": {
                    "dataset": "TestDataset2",
                    "index": 2,
                    "is_crop": 1,
                    "label": "maize",
                },
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
            },
        ],
    }


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    arrays_dir = os.path.join(root, "features", "arrays")
    os.makedirs(arrays_dir, exist_ok=True)

    for fname in ("0_TestDataset1.h5", "1_TestDataset1.h5", "2_TestDataset2.h5"):
        path = os.path.join(arrays_dir, fname)
        data = np.random.randint(0, 4000, size=(12, 18)).astype("int64")
        with h5py.File(path, "w") as f:
            f.create_dataset("array", data=data)

    with open(os.path.join(root, "labels.geojson"), "w") as f:
        json.dump(_make_geojson(), f)

    return root


def test_getitem(prepared_root):
    ds = CropHarvest(root=prepared_root, download=False)
    sample = ds[0]
    assert tuple(sample["array"].shape) == (12, 18)
    assert "label" in sample


def test_len(prepared_root):
    ds = CropHarvest(root=prepared_root, download=False)
    assert len(ds) == 3


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CropHarvest(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = CropHarvest(root=prepared_root, download=False)
    ds.plot(ds[0])
    plt.close()
