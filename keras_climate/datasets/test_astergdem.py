import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.astergdem import AsterGDEM


def _write_dem(path, ox, oy):
    transform = from_origin(ox, oy, 1, 1)
    data = np.random.randint(0, 1000, (1, 8, 8)).astype("int16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    _write_dem(os.path.join(str(tmp_path), "ASTGTMV003_N000000_dem.tif"), 0, 8)
    _write_dem(os.path.join(str(tmp_path), "ASTGTMV003_N000010_dem.tif"), 20, 8)
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = AsterGDEM(prepared_root)
    x = ds[ds.bounds]
    assert "mask" in x
    assert len(tuple(x["mask"].shape)) == 2


def test_len(prepared_root):
    ds = AsterGDEM(prepared_root)
    assert len(ds) == 2


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        AsterGDEM(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = AsterGDEM(prepared_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()


def test_plot_prediction(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from keras import ops

    ds = AsterGDEM(prepared_root)
    x = ds[ds.bounds]
    x["prediction"] = ops.convert_to_tensor(x["mask"])
    ds.plot(x, suptitle="Prediction")
    plt.close()
