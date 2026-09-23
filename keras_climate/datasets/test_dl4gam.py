import os

import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError  # noqa: E402
from keras_climate.datasets.dl4gam import DL4GAMAlps  # noqa: E402

PATCH_SIZE = 8
BAND_NAMES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
    "CLOUDLESS_MASK",
    "FILL_MASK",
]


def _create_sample(fp, rng):
    band_data = rng.randint(0, 10000, size=(15, PATCH_SIZE, PATCH_SIZE)).astype("int16")
    band_data[-2:] = (band_data[-2:] > 5000).astype("int16")

    data_dict = {
        "band_data": {
            "dims": ("band", "y", "x"),
            "data": band_data,
            "attrs": {"long_name": BAND_NAMES, "_FillValue": -9999},
        },
        "mask_all_g_id": {
            "dims": ("y", "x"),
            "data": rng.choice([-1, 8, 9], size=(PATCH_SIZE, PATCH_SIZE)).astype("int32"),
            "attrs": {"_FillValue": -1},
        },
        "mask_debris": {
            "dims": ("y", "x"),
            "data": (rng.random((PATCH_SIZE, PATCH_SIZE)) > 0.5).astype("int8"),
            "attrs": {"_FillValue": -1},
        },
    }
    for v in [
        "dem",
        "slope",
        "aspect",
        "planform_curvature",
        "profile_curvature",
        "terrain_ruggedness_index",
        "dhdt",
        "v",
    ]:
        data_dict[v] = {
            "dims": ("y", "x"),
            "data": (rng.random((PATCH_SIZE, PATCH_SIZE)) * 100).astype("float32"),
            "attrs": {"_FillValue": -9999},
        }

    nc = xr.Dataset.from_dict(data_dict)
    nc.to_netcdf(fp)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    rng = np.random.RandomState(42)

    glacier_ids = ["g_0008", "g_0009", "g_0030"]
    splits_df = pd.DataFrame(
        {
            "entry_id": glacier_ids,
            "split_1": ["fold_train", "fold_valid", "fold_test"],
        }
    )
    splits_df.to_csv(os.path.join(root, "splits.csv"), index=False)

    dir_small = os.path.join(root, "dataset_small")
    for glacier_id in glacier_ids:
        gdir = os.path.join(dir_small, glacier_id)
        os.makedirs(gdir, exist_ok=True)
        _create_sample(os.path.join(gdir, f"{glacier_id}_patch_0.nc"), rng)

    return root


def test_getitem(prepared_root):
    ds = DL4GAMAlps(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (PATCH_SIZE, PATCH_SIZE, 5)
    assert "mask_glacier" in sample
    assert "mask_debris" in sample
    assert "mask_clouds_and_shadows" in sample


def test_len(prepared_root):
    ds = DL4GAMAlps(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_extra_features(prepared_root):
    ds = DL4GAMAlps(root=prepared_root, split="train", extra_features=["dem", "slope"], download=False)
    sample = ds[0]
    assert "dem" in sample
    assert "slope" in sample


def test_invalid_split():
    with pytest.raises(AssertionError):
        DL4GAMAlps(split="bad")


def test_invalid_band():
    with pytest.raises(AssertionError):
        DL4GAMAlps(bands=("not_a_band",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        DL4GAMAlps(root=str(tmp_path), download=False)


def test_plot_rgb_missing_band(prepared_root):
    ds = DL4GAMAlps(root=prepared_root, split="train", bands=("B8",), download=False)
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DL4GAMAlps(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
