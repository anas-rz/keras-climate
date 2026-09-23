import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.gbm import GlobalBuildingMap


@pytest.fixture
def prepared_root(tmp_path):
    path = tmp_path / "GBM_v1_e000_n10_e005_n05.tif"
    transform = from_origin(0, 10, 1, 1)
    data = np.random.choice([0, 1], size=(1, 8, 8)).astype("uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = GlobalBuildingMap(paths=prepared_root)
    sample = ds[ds.bounds]
    assert "mask" in sample
    assert tuple(sample["mask"].shape) == (8, 8)


def test_len(prepared_root):
    ds = GlobalBuildingMap(paths=prepared_root)
    assert len(ds) == 1


def test_is_image_false(prepared_root):
    ds = GlobalBuildingMap(paths=prepared_root)
    assert ds.is_image is False
    assert ds.dtype == "int64"


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = GlobalBuildingMap(paths=prepared_root)
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()


def test_no_data(tmp_path):
    from keras_climate.datasets import DatasetNotFoundError

    with pytest.raises(DatasetNotFoundError):
        GlobalBuildingMap(paths=str(tmp_path))
