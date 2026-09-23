import contextlib
import io
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from keras_climate.datasets import meta_chm as meta_chm_module
from keras_climate.datasets.meta_chm import MetaCHM

SIZE = 8
RES = MetaCHM._res[0]
TILE = SIZE * RES


def _create_tile(tmp_path, quadkey, minx, miny, date):
    maxx, maxy = minx + TILE, miny + TILE
    directory = os.path.join(str(tmp_path), "chm")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{quadkey}.tif")
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "crs": "EPSG:3857",
        "transform": from_bounds(minx, miny, maxx, maxy, SIZE, SIZE),
        "height": SIZE,
        "width": SIZE,
    }
    data = np.random.randint(0, 30, size=(1, SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    return {
        "datetime": pd.Timestamp(date, tz="UTC"),
        "assets": {"chm": {"href": path}},
        "_box_3857": box(minx, miny, maxx, maxy),
    }


@pytest.fixture
def parquet_bytes(tmp_path):
    minx0, miny0 = 1490000.0, 6890000.0
    rows = [
        _create_tile(tmp_path, "tile1", minx0, miny0, "2019-08-09"),
        _create_tile(tmp_path, "tile2", minx0 + TILE, miny0, "2019-07-15"),
    ]
    gdf = gpd.GeoDataFrame(
        rows, geometry=[r.pop("_box_3857") for r in rows], crs="EPSG:3857"
    ).to_crs("EPSG:4326")

    parquet_path = os.path.join(str(tmp_path), "items.parquet")
    gdf.to_parquet(parquet_path)
    with open(parquet_path, "rb") as f:
        return f.read()


@pytest.fixture
def dataset(monkeypatch, parquet_bytes):
    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    @contextlib.contextmanager
    def fake_urlopen(request):
        yield _FakeResponse(parquet_bytes)

    monkeypatch.setattr(meta_chm_module.urllib.request, "urlopen", fake_urlopen)
    return MetaCHM()


def test_meta_chm_getitem(dataset):
    sample = dataset[dataset.bounds]
    assert "mask" in sample
    assert len(sample["mask"].shape) == 2


def test_meta_chm_len(dataset):
    assert len(dataset) == 2


def test_meta_chm_dtype_override(dataset):
    # MetaCHM overrides RasterDataset.dtype (normally a property returning
    # "int64" for is_image=False) with the plain attribute "float32", since
    # canopy height is a continuous value, not a class index.
    assert dataset.dtype == "float32"


def test_meta_chm_plot(dataset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = dataset[dataset.bounds]
    dataset.plot(sample, suptitle="Test")
    plt.close()
