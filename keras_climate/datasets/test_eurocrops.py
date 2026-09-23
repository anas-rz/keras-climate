import os

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.eurocrops import EuroCrops


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    gdf = gpd.GeoDataFrame(
        {
            "EC_hcat_c": ["1000000010", "9999999999"],
            "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)],
        },
        crs="EPSG:4326",
    )
    gdf.to_file(os.path.join(root, "AT_2021_EC.shp"))
    return root


def test_getitem(prepared_root):
    ds = EuroCrops(
        paths=prepared_root,
        res=(0.5, 0.5),
        classes=["1000000010"],
        download=False,
        checksum=False,
    )
    sample = ds[ds.bounds]
    assert "mask" in sample


def test_len(prepared_root):
    ds = EuroCrops(
        paths=prepared_root,
        res=(0.5, 0.5),
        classes=["1000000010"],
        download=False,
        checksum=False,
    )
    assert len(ds) == 1


def test_and_or(prepared_root):
    ds = EuroCrops(
        paths=prepared_root,
        res=(0.5, 0.5),
        classes=["1000000010"],
        download=False,
        checksum=False,
    )
    assert isinstance(ds & ds, IntersectionDataset)
    assert isinstance(ds | ds, UnionDataset)


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EuroCrops(paths=str(tmp_path), download=False, checksum=False)


def test_invalid_index(prepared_root):
    ds = EuroCrops(
        paths=prepared_root,
        res=(0.5, 0.5),
        classes=["1000000010"],
        download=False,
        checksum=False,
    )
    with pytest.raises(IndexError, match="not found in dataset with bounds"):
        ds[200:200, 200:200, pd.Timestamp.min : pd.Timestamp.min]


def test_get_label_with_none_hcat_code(prepared_root):
    ds = EuroCrops(
        paths=prepared_root,
        res=(0.5, 0.5),
        classes=["1000000010"],
        download=False,
        checksum=False,
    )
    label = ds.get_label(pd.Series({ds.label_name: None}))
    assert label == 0


def test_get_label_matches_class(prepared_root):
    ds = EuroCrops(
        paths=prepared_root,
        res=(0.5, 0.5),
        classes=["1000000010"],
        download=False,
        checksum=False,
    )
    assert ds.get_label(pd.Series({ds.label_name: "1000000010"})) == 1


def test_get_label_falls_back_to_zero(prepared_root):
    ds = EuroCrops(
        paths=prepared_root,
        res=(0.5, 0.5),
        classes=["1000000010"],
        download=False,
        checksum=False,
    )
    assert ds.get_label(pd.Series({ds.label_name: "9999999999"})) == 0


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = EuroCrops(
        paths=prepared_root,
        res=(0.5, 0.5),
        classes=["1000000010"],
        download=False,
        checksum=False,
    )
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
