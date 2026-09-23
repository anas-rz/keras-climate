import os

import fiona
import fiona.transform
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely.geometry
from rasterio.crs import CRS
from rasterio.transform import Affine

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.enviroatlas import EnviroAtlas

SUFFIX_TO_KEY = {
    "a_naip": ("naip", "continuous", 4, (4, 255)),
    "b_nlcd": ("nlcd", "categorical", 1, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]),
    "c_roads": ("roads", "categorical", 1, [0, 1]),
    "d_water": ("water", "categorical", 1, [0, 1]),
    "d1_waterways": ("waterways", "categorical", 1, [0, 1]),
    "d2_waterbodies": ("waterbodies", "categorical", 1, [0, 1]),
    "e_buildings": ("buildings", "categorical", 1, [0, 1]),
    "h_highres_labels": ("lc", "categorical", 1, [10, 20, 30, 40, 70]),
    "prior_from_cooccurrences_101_31": ("prior", "continuous", 5, (0, 225)),
    "prior_from_cooccurrences_101_31_no_osm_no_buildings": (
        "prior_no_osm_no_buildings",
        "continuous",
        5,
        (0, 220),
    ),
}


def _write_layer(path, data_type, count, vals, transform):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": count,
        "height": 16,
        "width": 16,
        "crs": CRS.from_epsg(26914),
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        size = (count, 16, 16)
        if data_type == "continuous":
            data = np.random.randint(vals[0], vals[1] + 1, size=size, dtype="uint8")
        else:
            data = np.random.choice(vals, size=size).astype("uint8")
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    folder = os.path.join(root, "enviroatlas_lotp")
    prefix = "pittsburgh_pa-2010_1m-train_tiles-debuffered/tile"
    transform = Affine(1.0, 0.0, 608170.0, 0.0, -1.0, 3381430.0)

    for suffix in SUFFIX_TO_KEY:
        _, data_type, count, vals = SUFFIX_TO_KEY[suffix]
        path = os.path.join(folder, f"{prefix}_{suffix}.tif")
        _write_layer(path, data_type, count, vals, transform)

    schema = {
        "geometry": "Polygon",
        "properties": {
            "split": "str",
            **{key: "str" for key, _, _, _ in SUFFIX_TO_KEY.values()},
        },
    }
    with fiona.open(
        os.path.join(folder, "spatial_index.geojson"),
        "w",
        driver="GeoJSON",
        crs="EPSG:3857",
        schema=schema,
    ) as dst:
        naip_path = os.path.join(folder, f"{prefix}_a_naip.tif")
        with rasterio.open(naip_path) as f:
            geom = shapely.geometry.mapping(shapely.geometry.box(*f.bounds))
            geom = fiona.transform.transform_geom(f.crs.to_string(), "EPSG:3857", geom)

        row = {
            "geometry": geom,
            "properties": {"split": "pittsburgh_pa-2010_1m-train"},
        }
        for suffix, (key, _, _, _) in SUFFIX_TO_KEY.items():
            row["properties"][key] = f"{prefix}_{suffix}.tif"
        dst.write(row)

    return root


@pytest.fixture
def dataset_cls(monkeypatch):
    monkeypatch.setattr(
        EnviroAtlas, "_files", ["pittsburgh_pa-2010_1m-train_tiles-debuffered", "spatial_index.geojson"]
    )
    return EnviroAtlas


def test_getitem(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, layers=["naip", "prior", "lc"], download=False)
    x = ds[ds.bounds]
    assert isinstance(x, dict)
    assert tuple(x["image"].shape[-1:]) == (4,)
    # NOTE: upstream torchgeo derives the "prior"/"prior_no_osm_no_buildings"
    # columns via `gdf['naip'].replace('a_naip', ...)`, which is a pandas
    # Series.replace() exact-value match (not a substring replace), so it is
    # always a no-op against real "*_a_naip.tif" filenames. This means the
    # "prior" layer actually re-reads the naip file (4 bands here), not the
    # dedicated prior file (5 bands). We faithfully preserve this quirk
    # rather than "fixing" it, so mask = prior-via-naip (4 bands) + lc
    # (1 band) = 5 channels.
    assert tuple(x["mask"].shape[-1:]) == (5,)


def test_len(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, download=False)
    assert len(ds) == 1


def test_and_or(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, download=False)
    assert isinstance(ds & ds, IntersectionDataset)
    assert isinstance(ds | ds, UnionDataset)


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EnviroAtlas(root=str(tmp_path), download=False)


def test_out_of_bounds_index(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, download=False)
    with pytest.raises(IndexError, match="not found in dataset with bounds"):
        ds[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]


def test_invalid_split_raises(dataset_cls, prepared_root):
    with pytest.raises(AssertionError):
        dataset_cls(root=prepared_root, splits=["not-a-real-split"], download=False)


def test_invalid_layer_raises(dataset_cls, prepared_root):
    with pytest.raises(AssertionError):
        dataset_cls(root=prepared_root, layers=["not-a-real-layer"], download=False)


def test_plot(dataset_cls, prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = dataset_cls(root=prepared_root, layers=["naip", "prior", "lc"], download=False)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()


def test_plot_missing_layers_raises(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, layers=["naip", "prior"], download=False)
    x = ds[ds.bounds]
    with pytest.raises(ValueError, match="The 'naip' and"):
        ds.plot(x)
