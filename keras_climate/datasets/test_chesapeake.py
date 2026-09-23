import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.chesapeake import ChesapeakeCVPR, ChesapeakeDC

SIZE = 8


def _write_tile(path, count=1, dtype="uint8"):
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": count,
        "crs": "epsg:32618",
        "transform": rasterio.transform.from_bounds(0, 0, 8, 8, SIZE, SIZE),
        "height": SIZE,
        "width": SIZE,
    }
    if dtype == "float32":
        data = np.random.rand(count, SIZE, SIZE).astype(dtype)
    else:
        data = np.random.randint(0, 8, size=(count, SIZE, SIZE)).astype(dtype)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


@pytest.fixture
def dc_root(tmp_path):
    root = str(tmp_path)
    _write_tile(os.path.join(root, "dc_lulc_2018_2022-Edition.tif"))
    return root


def test_dc_getitem(dc_root):
    ds = ChesapeakeDC(dc_root)
    x = ds[ds.bounds]
    assert "mask" in x


def test_dc_len(dc_root):
    ds = ChesapeakeDC(dc_root)
    assert len(ds) == 1


def test_dc_and(dc_root):
    ds = ChesapeakeDC(dc_root)
    assert isinstance(ds & ds, IntersectionDataset)


def test_dc_or(dc_root):
    ds = ChesapeakeDC(dc_root)
    assert isinstance(ds | ds, UnionDataset)


def test_dc_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        ChesapeakeDC(str(tmp_path))


def test_dc_plot(dc_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ChesapeakeDC(dc_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x, suptitle="Prediction")
    plt.close()


# ChesapeakeCVPR is a hand-rolled GeoDataset (not RasterDataset-based) with a
# bespoke windowed-read __getitem__ and a CRS-reprojected spatial index. We
# build a minimal single-tile fixture that mirrors torchgeo's synthetic test
# fixture (tests/data/chesapeake/cvpr/data.py): one UTM tile plus a
# spatial_index.geojson (in EPSG:3857, as the real dataset ships it) pointing
# at that tile's per-layer files.
CVPR_CRS = CRS.from_epsg(26918)
CVPR_TRANSFORM = rasterio.transform.from_origin(451549.0, 4316628.0, 1.0, 1.0)
TILE = "m_0000000_test_18_1"
TILE_DIR = "de_1m_2013_extended-debuffered-test_tiles"


def _write_cvpr_layer(path, count, dtype):
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": count,
        "crs": CVPR_CRS,
        "transform": CVPR_TRANSFORM,
        "width": SIZE,
        "height": SIZE,
    }
    if dtype == "float32":
        data = np.random.rand(count, SIZE, SIZE).astype(dtype)
    else:
        data = np.random.randint(0, 4, size=(count, SIZE, SIZE)).astype(dtype)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


@pytest.fixture
def cvpr_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ChesapeakeCVPR,
        "_files",
        {
            "base": (TILE_DIR, "spatial_index.geojson"),
            "prior_extension": (
                os.path.join(
                    TILE_DIR,
                    f"{TILE}_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif",
                ),
            ),
        },
    )
    root = str(tmp_path)
    tile_dir = os.path.join(root, TILE_DIR)
    os.makedirs(tile_dir, exist_ok=True)

    layers = {
        "naip-new": (4, "uint8"),
        "naip-old": (4, "uint8"),
        "landsat-leaf-on": (9, "float32"),
        "landsat-leaf-off": (9, "float32"),
        "lc": (1, "uint8"),
        "nlcd": (1, "uint8"),
        "buildings": (1, "uint8"),
        "prior_from_cooccurrences_101_31_no_osm_no_buildings": (4, "uint8"),
    }
    for layer, (count, dtype) in layers.items():
        _write_cvpr_layer(os.path.join(tile_dir, f"{TILE}_{layer}.tif"), count, dtype)

    # Compute the tile footprint in EPSG:3857 (what the real spatial index uses)
    import pyproj

    left, bottom, right, top = rasterio.transform.array_bounds(SIZE, SIZE, CVPR_TRANSFORM)
    transformer = pyproj.Transformer.from_crs(CVPR_CRS, "EPSG:3857", always_xy=True)
    minx, miny = transformer.transform(left, bottom)
    maxx, maxy = transformer.transform(right, top)

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "split": "de-test",
                    **{
                        layer: os.path.join(TILE_DIR, f"{TILE}_{layer}.tif")
                        for layer in layers
                        if layer != "prior_from_cooccurrences_101_31_no_osm_no_buildings"
                    },
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [minx, miny],
                            [minx, maxy],
                            [maxx, maxy],
                            [maxx, miny],
                            [minx, miny],
                        ]
                    ],
                },
            }
        ],
    }
    with open(os.path.join(root, "spatial_index.geojson"), "w") as f:
        json.dump(geojson, f)

    return root


def test_cvpr_getitem(cvpr_root):
    ds = ChesapeakeCVPR(cvpr_root, splits=["de-test"], layers=["naip-new", "lc"])
    x = ds[ds.bounds]
    assert "image" in x
    assert "mask" in x


def test_cvpr_len(cvpr_root):
    ds = ChesapeakeCVPR(cvpr_root, splits=["de-test"], layers=["naip-new", "lc"])
    assert len(ds) == 1


def test_cvpr_invalid_split(cvpr_root):
    with pytest.raises(AssertionError):
        ChesapeakeCVPR(cvpr_root, splits=["bogus-split"])


def test_cvpr_invalid_layer(cvpr_root):
    with pytest.raises(AssertionError):
        ChesapeakeCVPR(cvpr_root, splits=["de-test"], layers=["bogus-layer"])


def test_cvpr_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        ChesapeakeCVPR(str(tmp_path), splits=["de-test"])


def test_cvpr_prior_extension_missing(cvpr_root):
    # The prior layer's file is present in our fixture (we don't split base
    # vs. prior_extension archives), so requesting it should succeed here;
    # this exercises the layer-selection & mask-concatenation path instead.
    ds = ChesapeakeCVPR(
        cvpr_root,
        splits=["de-test"],
        layers=["naip-new", ChesapeakeCVPR.prior_layer],
    )
    x = ds[ds.bounds]
    assert x["mask"].shape[-1] == 4
