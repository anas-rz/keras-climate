import json
import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

pytest.importorskip("pyarrow")

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets._test_helpers import assert_dtype, assert_int64_dtype
from keras_climate.datasets.bigearthnet import BigEarthNet, BigEarthNetV2

S2_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)


def _write_tif(path, count=1, size=4):
    transform = from_origin(0, size, 1, 1)
    data = np.random.randint(0, 255, (count, size, size)).astype("uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=count,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def v1_root(tmp_path):
    root = str(tmp_path)
    s1_patch = "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39"
    s2_patch = "S2A_MSIL2A_20170613T101031_61_39"

    s1_dir = os.path.join(root, "BigEarthNet-S1-v1.0", s1_patch)
    s2_dir = os.path.join(root, "BigEarthNet-v1.0", s2_patch)
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)

    for pol in ("VV", "VH"):
        _write_tif(os.path.join(s1_dir, f"{s1_patch}_{pol}.tif"))
    with open(os.path.join(s1_dir, f"{s1_patch}_labels_metadata.json"), "w") as f:
        json.dump({"labels": ["Continuous urban fabric"]}, f)

    for band in S2_BANDS:
        _write_tif(os.path.join(s2_dir, f"{s2_patch}_{band}.tif"))
    with open(os.path.join(s2_dir, f"{s2_patch}_labels_metadata.json"), "w") as f:
        json.dump({"labels": ["Continuous urban fabric"]}, f)

    for split in ("train", "val", "test"):
        filename = BigEarthNet.splits_metadata[split]["filename"]
        with open(os.path.join(root, filename), "w") as f:
            f.write(f"{s2_patch},{s1_patch}\n")

    return root


@pytest.mark.parametrize("bands,num_classes", [("all", 43), ("s1", 19), ("s2", 19)])
def test_getitem_v1(v1_root, bands, num_classes):
    ds = BigEarthNet(v1_root, split="train", bands=bands, num_classes=num_classes)
    x = ds[0]
    assert_dtype(x["image"], "float32")
    assert_int64_dtype(x["label"])
    assert x["label"].shape == (num_classes,)
    if bands == "all":
        assert x["image"].shape == (120, 120, 14)
    elif bands == "s1":
        assert x["image"].shape == (120, 120, 2)
    else:
        assert x["image"].shape == (120, 120, 12)


def test_len_v1(v1_root):
    ds = BigEarthNet(v1_root, split="train")
    assert len(ds) == 1


def test_not_downloaded_v1(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        BigEarthNet(str(tmp_path))


def test_plot_v1(v1_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = BigEarthNet(v1_root, split="train")
    x = ds[0].copy()
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["label"]
    ds.plot(x, show_titles=False)
    plt.close()


@pytest.fixture
def v2_root(tmp_path):
    root = str(tmp_path)
    s1_name = "S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39"
    patch_id = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
    s2_patch_dir = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP"
    s1_patch_dir = "S1A_IW_GRDH_1SDV_20170613T165043"

    s1_dir = os.path.join(root, "BigEarthNet-S1", s1_patch_dir, s1_name)
    s2_dir = os.path.join(root, "BigEarthNet-S2", s2_patch_dir, patch_id)
    maps_dir = os.path.join(root, "Reference_Maps", s2_patch_dir, patch_id)
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    for pol in ("VV", "VH"):
        _write_tif(os.path.join(s1_dir, f"{s1_name}_{pol}.tif"))
    for band in S2_BANDS:
        _write_tif(os.path.join(s2_dir, f"{patch_id}_{band}.tif"))

    clc_code = 111  # maps to ordinal 0 (Urban fabric)
    map_path = os.path.join(maps_dir, f"{patch_id}_reference_map.tif")
    transform = from_origin(0, 4, 1, 1)
    with rasterio.open(
        map_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="int32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.full((4, 4), clc_code, dtype="int32"), 1)

    df = pd.DataFrame(
        {
            "patch_id": [patch_id],
            "s1_name": [s1_name],
            "split": ["train"],
            "labels": [["Urban fabric"]],
        }
    )
    df.to_parquet(os.path.join(root, "metadata.parquet"))

    return root


@pytest.mark.parametrize("bands", ["all", "s1", "s2"])
def test_getitem_v2(v2_root, bands):
    ds = BigEarthNetV2(v2_root, split="train", bands=bands)
    x = ds[0]
    if bands in ("s2", "all"):
        key = "image_s2" if bands == "all" else "image"
        if bands in ("s2", "all"):
            assert x[key].shape == (120, 120, 12)
    if bands in ("s1", "all"):
        key = "image_s1" if bands == "all" else "image"
        assert x[key].shape == (120, 120, 2)
    assert x["mask"].shape == (4, 4, 1)
    assert_int64_dtype(x["mask"])
    assert_int64_dtype(x["label"])


def test_len_v2(v2_root):
    ds = BigEarthNetV2(v2_root, split="train")
    assert len(ds) == 1


def test_not_downloaded_v2(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        BigEarthNetV2(str(tmp_path))


def test_plot_v2(v2_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = BigEarthNetV2(v2_root, split="train", bands="all")
    x = ds[0].copy()
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["label"]
    ds.plot(x, show_titles=False)
    plt.close()
