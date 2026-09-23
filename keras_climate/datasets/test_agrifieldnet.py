import os

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from keras_climate.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    UnionDataset,
)
from keras_climate.datasets.agrifieldnet import (
    AgriFieldNet,
    AgriFieldNetImage,
    AgriFieldNetMask,
)

SIZE = 8


def _profile(count):
    return {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": SIZE,
        "height": SIZE,
        "count": count,
        "crs": CRS.from_epsg(32644),
        "transform": Affine(10.0, 0.0, 535840.0, 0.0, -10.0, 3079680.0),
    }


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    folder_id = "00001"

    source_dir = os.path.join(
        root, "source", f"ref_agrifieldnet_competition_v1_source_{folder_id}"
    )
    os.makedirs(source_dir, exist_ok=True)
    for band in AgriFieldNetImage.all_bands:
        path = os.path.join(
            source_dir,
            f"ref_agrifieldnet_competition_v1_source_{folder_id}_{band}_10m.tif",
        )
        data = np.random.randint(0, 255, (SIZE, SIZE), dtype="uint8")
        with rasterio.open(path, "w", **_profile(1)) as dst:
            dst.write(data, 1)

    labels_dir = os.path.join(root, "train_labels")
    os.makedirs(labels_dir, exist_ok=True)
    mask_path = os.path.join(
        labels_dir, f"ref_agrifieldnet_competition_v1_labels_train_{folder_id}.tif"
    )
    mask_data = np.random.choice(
        list(AgriFieldNetMask.valid_classes), size=(SIZE, SIZE)
    ).astype("uint8")
    with rasterio.open(mask_path, "w", **_profile(1)) as dst:
        dst.write(mask_data, 1)

    return root


def test_getitem(prepared_root):
    ds = AgriFieldNet(prepared_root)
    x = ds[ds.bounds]
    assert "image" in x
    assert "mask" in x
    assert x["image"].shape[-1] == len(AgriFieldNetImage.all_bands)


def test_len(prepared_root):
    ds = AgriFieldNet(prepared_root)
    assert len(ds) == 1


def test_and(prepared_root):
    ds = AgriFieldNet(prepared_root)
    assert isinstance(ds & ds, IntersectionDataset)


def test_or(prepared_root):
    ds = AgriFieldNet(prepared_root)
    assert isinstance(ds | ds, UnionDataset)


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        AgriFieldNet(str(tmp_path))


def test_invalid_classes(prepared_root):
    with pytest.raises(AssertionError):
        AgriFieldNet(prepared_root, classes=[1])  # missing background class 0


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = AgriFieldNet(prepared_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()


def test_rgb_bands_missing_plot(prepared_root):
    ds = AgriFieldNet(prepared_root, bands=["B01", "B05", "B06"])
    x = ds[ds.bounds]
    with pytest.raises(RGBBandsMissingError):
        ds.plot(x)
