import json
import os
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.mmearth import MMEarth

PX_DIM = (4, 4)
NUM_TILES = 3

ALL_MODALITY_BANDS = MMEarth.all_modality_bands

MODALITIES = {
    "sentinel2": {"bands": 13, "dtype": np.uint16},
    "aster": {"bands": 2, "dtype": np.int16},
    "era5": {"bands": 12, "dtype": np.float32},
    "esa_worldcover": {"bands": 1, "dtype": np.uint8},
    "biome": {"bands": 14, "dtype": np.uint8},
}

META_DUMMY = {
    "S2_DATE": "2018-07-16",
    "S2_type": "l1c",
    "CRS": "EPSG:32721",
    "lat": -14.5,
    "lon": -56.9,
}


@pytest.fixture
def mmearth_root(tmp_path):
    np.random.seed(0)
    dirname = "data_1M_v001_64"
    base = str(tmp_path)
    data_dir = os.path.join(base, dirname)
    os.makedirs(data_dir)

    tile_info = {}
    with h5py.File(os.path.join(data_dir, f"{dirname}.h5"), "w") as h5file:
        for modality, info in MODALITIES.items():
            bands = info["bands"]
            if modality in ("era5", "biome"):
                h5file.create_dataset(modality, (NUM_TILES, bands), dtype=info["dtype"])
            else:
                h5file.create_dataset(
                    modality, (NUM_TILES, bands, *PX_DIM), dtype=info["dtype"]
                )
        h5file.create_dataset(
            "metadata", (NUM_TILES,), dtype=np.dtype([("meta_id", "S10"), ("S2_type", "S3")])
        )

        for i in range(NUM_TILES):
            for modality, info in MODALITIES.items():
                bands = info["bands"]
                if modality == "sentinel2":
                    data = np.random.randint(0, 65535, size=(bands, *PX_DIM))
                elif modality == "esa_worldcover":
                    data = np.random.choice(
                        [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255],
                        size=(bands, *PX_DIM),
                    )
                elif modality == "aster":
                    data = np.random.random(size=(bands, *PX_DIM)) * 100
                elif modality == "era5":
                    data = np.random.random(size=(bands,))
                elif modality == "biome":
                    data = np.random.randint(0, 2, size=(bands,))
                h5file[modality][i] = data.astype(info["dtype"])

            S2_type = np.random.choice(["l1c", "l2a"]).encode("utf-8")
            meta_id = str(i).encode("utf-8")
            h5file["metadata"][i] = (meta_id, S2_type)

            tile_meta = deepcopy(META_DUMMY)
            tile_meta["S2_type"] = S2_type.decode("utf-8")
            date_obj = pd.to_datetime(tile_meta["S2_DATE"], format="%Y-%m-%d")
            curr_month_str = date_obj.strftime("%Y%m")
            prev_month_obj = date_obj.replace(day=1) - pd.Timedelta(days=1)
            prev_month_str = prev_month_obj.strftime("%Y%m")
            sample_bands = deepcopy(ALL_MODALITY_BANDS)
            sample_bands["era5"] = [
                b.replace("curr", curr_month_str).replace("prev", prev_month_str)
                for b in sample_bands["era5"]
            ]
            tile_meta["BANDS"] = sample_bands
            tile_info[str(i)] = tile_meta

    band_modalities = {
        "sentinel2_l1c": 13,
        "sentinel2_l2a": 13,
        "aster": 2,
        "era5": 12,
    }
    band_stats = {
        modality: {
            "mean": np.random.random(size=(n,)).tolist(),
            "std": (np.random.random(size=(n,)) + 0.1).tolist(),
            "min": np.zeros(n).tolist(),
            "max": (np.random.random(size=(n,)) + 1).tolist(),
        }
        for modality, n in band_modalities.items()
    }

    splits = {"train": list(range(NUM_TILES)), "val": [], "test": []}

    with open(os.path.join(data_dir, f"{dirname}_splits.json"), "w") as f:
        json.dump(splits, f)
    with open(os.path.join(data_dir, f"{dirname}_band_stats.json"), "w") as f:
        json.dump(band_stats, f)
    with open(os.path.join(data_dir, f"{dirname}_tile_info.json"), "w") as f:
        json.dump(tile_info, f)

    return base


def test_mmearth_getitem(mmearth_root):
    ds = MMEarth(
        root=mmearth_root,
        subset="MMEarth64",
        modalities=["sentinel2", "aster", "era5", "esa_worldcover", "biome"],
    )
    sample = ds[0]
    assert tuple(sample["image_sentinel2"].shape) == (4, 4, 13)
    assert tuple(sample["image_aster"].shape) == (4, 4, 2)
    assert tuple(sample["mask_esa_worldcover"].shape) == (4, 4, 1)
    assert "era5" in sample
    assert "biome" in sample
    assert "lat" in sample
    assert "lon" in sample
    assert "date" in sample
    assert "tile_id" in sample


def test_mmearth_len(mmearth_root):
    ds = MMEarth(root=mmearth_root, subset="MMEarth64", modalities=["sentinel2"])
    assert len(ds) == NUM_TILES


def test_mmearth_invalid_subset(mmearth_root):
    with pytest.raises(AssertionError):
        MMEarth(root=mmearth_root, subset="bogus")


def test_mmearth_invalid_modality(mmearth_root):
    with pytest.raises(ValueError):
        MMEarth(root=mmearth_root, subset="MMEarth64", modalities=["bogus"])


def test_mmearth_invalid_modality_band(mmearth_root):
    with pytest.raises(ValueError):
        MMEarth(
            root=mmearth_root,
            subset="MMEarth64",
            modalities=["sentinel2"],
            modality_bands={"sentinel2": ["bogus"]},
        )


def test_mmearth_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        MMEarth(root=str(tmp_path), subset="MMEarth64", modalities=["sentinel2"])


def test_mmearth_min_max_normalization(mmearth_root):
    ds = MMEarth(
        root=mmearth_root,
        subset="MMEarth64",
        modalities=["sentinel2"],
        normalization_mode="min-max",
    )
    sample = ds[0]
    assert tuple(sample["image_sentinel2"].shape) == (4, 4, 13)
