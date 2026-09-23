import os

import numpy as np
import pytest
import rasterio
from rasterio import Affine

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.nasa_marine_debris import NASAMarineDebris

SIZE = 16


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "source"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels"), exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 3,
        "crs": "epsg:4326",
        "transform": Affine(2e-5, 0.0, -87.6, 0.0, -2e-5, 15.9),
        "height": SIZE,
        "width": SIZE,
    }

    for i in range(3):
        name = f"scene_{i}"
        data = np.random.randint(0, 255, size=(3, SIZE, SIZE)).astype("uint8")
        with rasterio.open(os.path.join(root, "source", f"{name}.tif"), "w", **profile) as dst:
            dst.write(data)

        # include one invalid (zero-area) box to exercise the filtering logic
        boxes = np.array(
            [[1, 1, 5, 5, 1], [2, 2, 2, 6, 1], [3, 3, 10, 10, 1]], dtype="float32"
        )
        np.save(os.path.join(root, "labels", f"{name}.npy"), boxes)

    return root


def test_getitem(prepared_root):
    ds = NASAMarineDebris(root=prepared_root)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (SIZE, SIZE, 3)
    # the zero-width box should have been filtered out
    assert tuple(sample["bbox_xyxy"].shape) == (2, 4)


def test_len(prepared_root):
    ds = NASAMarineDebris(root=prepared_root)
    assert len(ds) == 3


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        NASAMarineDebris(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = NASAMarineDebris(root=prepared_root)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()
    sample["prediction_bbox_xyxy"] = sample["bbox_xyxy"]
    ds.plot(sample)
    plt.close()
