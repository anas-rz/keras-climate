import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.flair import TASKS, FLAIRHUB, FLAIRHUBBase, FLAIRHUBToy

# FLAIRHUB's real _verify() iterates over its full 70-domain `domain_years`
# table (checking directories for every domain/year/modality combination),
# so a full-fidelity fixture is impractical. We monkeypatch `domain_years`
# down to a single domain/year and request a single band so only a tiny
# on-disk layout is needed, while still exercising the real _verify(),
# _load_files(), __getitem__, and plot() code paths end to end.


def _write_tif(path, count, transform):
    data = np.random.randint(0, 255, (count, 8, 8)).astype("uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=count,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path, monkeypatch):
    monkeypatch.setattr(FLAIRHUBBase, "domain_years", {"D004": ["2021"]})

    root = str(tmp_path)
    transform = from_origin(0, 8, 1, 1)

    label_dir = os.path.join(root, "D004-2021_AERIAL_LABEL-COSIA", "t1")
    rgbi_dir = os.path.join(root, "D004-2021_AERIAL_RGBI", "t1")
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(rgbi_dir, exist_ok=True)

    _write_tif(
        os.path.join(label_dir, "AERIAL_LABEL-COSIA_D004-2021_t1_p1.tif"), 1, transform
    )
    _write_tif(os.path.join(rgbi_dir, "AERIAL_RGBI_D004-2021_t1_p1.tif"), 4, transform)

    splits_dir = os.path.join(root, "GLOBAL_ALL_MTD")
    os.makedirs(splits_dir, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {
            "patch_id": ["D004-2021_t1_p1"],
            "split_1": ["train"],
            "geometry": [Point(0, 0)],
        },
        crs="EPSG:4326",
    )
    gdf.to_file(os.path.join(splits_dir, "GLOBAL_ALL_MTD_SPLIT.gpkg"), driver="GPKG")

    return root


def test_getitem(prepared_root):
    ds = FLAIRHUBBase(
        root=prepared_root, split="train", bands=["AERIAL_RGBI"], download=False, checksum=False
    )
    sample = ds[0]
    assert tuple(sample["mask"].shape) == (8, 8)
    assert tuple(sample["image_aerial_rgbi"].shape) == (8, 8, 4)


def test_len(prepared_root):
    ds = FLAIRHUBBase(
        root=prepared_root, split="train", bands=["AERIAL_RGBI"], download=False, checksum=False
    )
    assert len(ds) == 1


def test_invalid_band(prepared_root):
    with pytest.raises(ValueError):
        FLAIRHUBBase(root=prepared_root, bands=["NOT_A_BAND"], download=False, checksum=False)


def test_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(FLAIRHUBBase, "domain_years", {"D004": ["2021"]})
    with pytest.raises(DatasetNotFoundError):
        FLAIRHUBBase(
            root=str(tmp_path), bands=["AERIAL_RGBI"], download=False, checksum=False
        )


def test_tasks_classes():
    # band/class-list invariants
    assert len(TASKS["land_cover"]["classes"]) == 19
    assert len(TASKS["crop_type"]["classes"]) == 23


def test_flairhub_is_subclass():
    assert issubclass(FLAIRHUB, FLAIRHUBBase)
    assert issubclass(FLAIRHUBToy, FLAIRHUBBase)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = FLAIRHUBBase(
        root=prepared_root, split="train", bands=["AERIAL_RGBI"], download=False, checksum=False
    )
    ds.plot(ds[0], suptitle="Test")
    plt.close()
