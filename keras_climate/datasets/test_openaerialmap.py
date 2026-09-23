import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.openaerialmap import OpenAerialMap, TileUtils


def _write_tile(root, x, y, z):
    minx, miny, maxx, maxy = 85.516, 27.631, 85.523, 27.637
    height, width = 32, 32
    data = np.random.randint(0, 255, size=(3, height, width), dtype="uint8")
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    path = os.path.join(root, f"OAM-{x}-{y}-{z}.tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    _write_tile(root, 372608, 213968, 19)
    _write_tile(root, 372609, 213968, 19)
    return root


def test_getitem(prepared_root):
    ds = OpenAerialMap(paths=prepared_root)
    x = ds[ds.bounds]
    assert x["image"].shape[-1] == 3


def test_len(prepared_root):
    ds = OpenAerialMap(paths=prepared_root)
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        OpenAerialMap(paths=str(tmp_path))


def test_download_requires_bbox(tmp_path):
    with pytest.raises(ValueError, match="bbox must be provided"):
        OpenAerialMap(paths=str(tmp_path), download=True)


def test_download_invalid_zoom(tmp_path):
    with pytest.raises(ValueError, match="zoom must be between"):
        OpenAerialMap(
            paths=str(tmp_path), download=True, bbox=(0, 0, 1, 1), zoom=99
        )


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = OpenAerialMap(paths=prepared_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()


def test_tile_utils_roundtrip():
    tile = TileUtils.tile(85.52, 27.63, 19)
    bounds = TileUtils.bounds(tile)
    assert bounds.west < 85.52 < bounds.east
    assert bounds.south < 27.63 < bounds.north


def test_tile_utils_tiles_generator():
    tiles = list(TileUtils.tiles(85.516, 27.631, 85.523, 27.637, 19, truncate=True))
    assert len(tiles) > 0
    assert all(isinstance(t, TileUtils.Tile) for t in tiles)
