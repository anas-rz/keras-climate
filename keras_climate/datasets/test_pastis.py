import os

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.pastis import PASTIS, PASTIS100

SIZE = 8
T = 3


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    base = os.path.join(root, "PASTIS-R")
    for sub in ("DATA_S2", "DATA_S1A", "DATA_S1D", "ANNOTATIONS", "INSTANCE_ANNOTATIONS"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    num_samples = 3
    for i in range(num_samples):
        np.save(
            os.path.join(base, "DATA_S2", f"S2_{i}.npy"),
            np.random.randint(0, 256, size=(T, 10, SIZE, SIZE)).astype("int16"),
        )
        np.save(
            os.path.join(base, "DATA_S1A", f"S1A_{i}.npy"),
            np.random.rand(T, 3, SIZE, SIZE).astype("float32"),
        )
        np.save(
            os.path.join(base, "DATA_S1D", f"S1D_{i}.npy"),
            np.random.rand(T, 3, SIZE, SIZE).astype("float32"),
        )
        np.save(
            os.path.join(base, "ANNOTATIONS", f"TARGET_{i}.npy"),
            np.random.randint(0, 20, size=(3, SIZE, SIZE)).astype("uint8"),
        )
        np.save(
            os.path.join(base, "INSTANCE_ANNOTATIONS", f"INSTANCES_{i}.npy"),
            np.random.randint(0, 4, size=(SIZE, SIZE)).astype("int64"),
        )

    polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
    gdf = gpd.GeoDataFrame(
        {
            "Fold": [(i % 5) + 1 for i in range(num_samples)],
            "ID_PATCH": list(range(num_samples)),
            "geometry": [polygon] * num_samples,
        },
        crs="EPSG:4326",
    )
    gdf.to_file(os.path.join(base, "metadata.geojson"), driver="GeoJSON")

    return root


def test_getitem_semantic(prepared_root):
    ds = PASTIS(root=prepared_root, mode="semantic", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (T, SIZE, SIZE, 10)
    assert tuple(sample["mask"].shape) == (SIZE, SIZE)


def test_getitem_instance(prepared_root):
    ds = PASTIS(root=prepared_root, mode="instance", download=False)
    sample = ds[0]
    assert "bbox_xyxy" in sample
    assert "label" in sample
    assert sample["mask"].shape[-2:] == (SIZE, SIZE)


def test_len(prepared_root):
    ds = PASTIS(root=prepared_root, download=False)
    assert len(ds) == 3


def test_folds(prepared_root):
    ds = PASTIS(root=prepared_root, folds=(1,), download=False)
    assert len(ds) <= 3


def test_band_subset(prepared_root):
    ds = PASTIS(root=prepared_root, bands=("S1A_VV", "S1A_VH"), download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (T, SIZE, SIZE, 2)


def test_invalid_bands(prepared_root):
    with pytest.raises(ValueError):
        PASTIS(root=prepared_root, bands=("BOGUS",), download=False)


def test_invalid_fold(prepared_root):
    with pytest.raises(AssertionError):
        PASTIS(root=prepared_root, folds=(9,), download=False)


def test_invalid_mode(prepared_root):
    with pytest.raises(AssertionError):
        PASTIS(root=prepared_root, mode="bogus", download=False)


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        PASTIS(root=str(tmp_path), download=False)


def test_plot_semantic(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = PASTIS(root=prepared_root, mode="semantic", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()


def test_pastis100_defaults():
    assert PASTIS100.directory == "PASTIS-R"
    assert PASTIS100.md5 != PASTIS.md5
