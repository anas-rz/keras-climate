import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.solar_plants_brazil import SolarPlantsBrazil


def _write_tif(path, size=8, count=4, dtype="float32", value=0.5):
    transform = from_origin(0, size, 1, 1)
    data = np.full((count, size, size), value, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    split_dir = os.path.join(root, "train")
    input_dir = os.path.join(split_dir, "input")
    labels_dir = os.path.join(split_dir, "labels")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for i in range(2):
        _write_tif(os.path.join(input_dir, f"img({i}).tif"))
        _write_tif(
            os.path.join(labels_dir, f"target({i}).tif"),
            count=1,
            dtype="uint8",
            value=1 if i == 0 else 0,
        )
    return root


def test_getitem(prepared_root):
    ds = SolarPlantsBrazil(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 4)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(prepared_root):
    ds = SolarPlantsBrazil(root=prepared_root, split="train", download=False)
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SolarPlantsBrazil(root=str(tmp_path), split="train", download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SolarPlantsBrazil(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()


def test_plot_prediction(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SolarPlantsBrazil(root=prepared_root, split="train", download=False)
    sample = ds[0]
    sample["prediction"] = sample["mask"]
    ds.plot(sample)
    plt.close()
