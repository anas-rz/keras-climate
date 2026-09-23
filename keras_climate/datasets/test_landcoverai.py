import os

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.landcoverai import LandCoverAI, LandCoverAI100, LandCoverAIGeo


@pytest.fixture
def geo_root(tmp_path):
    base = str(tmp_path)
    transform = from_origin(0, 8, 1, 1)
    os.makedirs(os.path.join(base, "images"))
    os.makedirs(os.path.join(base, "masks"))
    img = np.random.randint(0, 255, (3, 8, 8)).astype("uint8")
    mask = np.random.randint(0, 5, (1, 8, 8)).astype("uint8")
    with rasterio.open(
        os.path.join(base, "images", "tile.tif"),
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(img)
    with rasterio.open(
        os.path.join(base, "masks", "tile.tif"),
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(mask)
    return base


def test_landcoverai_geo_getitem(geo_root):
    ds = LandCoverAIGeo(geo_root)
    sample = ds[ds.bounds]
    assert tuple(sample["image"].shape[-1:]) == (3,)
    assert "mask" in sample
    assert len(sample["mask"].shape) == 2


def test_landcoverai_geo_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        LandCoverAIGeo(str(tmp_path), download=False)


@pytest.fixture(params=[LandCoverAI, LandCoverAI100])
def nongeo_cls(request):
    return request.param


@pytest.fixture
def nongeo_root(tmp_path, nongeo_cls):
    base = str(tmp_path)
    os.makedirs(os.path.join(base, "output"))
    ids = ["tile_0", "tile_1"]
    for split in ["train", "val", "test"]:
        with open(os.path.join(base, f"{split}.txt"), "w") as f:
            f.write("\n".join(ids) + "\n")
    for id_ in ids:
        img = (np.random.rand(8, 8, 3) * 255).astype("uint8")
        Image.fromarray(img).save(os.path.join(base, "output", f"{id_}.jpg"))
        mask = np.random.randint(0, 5, (8, 8)).astype("uint8")
        Image.fromarray(mask).save(os.path.join(base, "output", f"{id_}_m.png"))
    return base


def test_landcoverai_nongeo_getitem(nongeo_cls, nongeo_root):
    ds = nongeo_cls(root=nongeo_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)
    assert tuple(sample["mask"].shape) == (8, 8)


def test_landcoverai_nongeo_len(nongeo_cls, nongeo_root):
    ds = nongeo_cls(root=nongeo_root, split="train", download=False)
    assert len(ds) == 2


def test_landcoverai_nongeo_invalid_split(nongeo_root):
    with pytest.raises(AssertionError):
        LandCoverAI(root=nongeo_root, split="bogus")


def test_landcoverai_nongeo_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        LandCoverAI(root=str(tmp_path), download=False)


def test_landcoverai_plot(nongeo_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = LandCoverAI(root=nongeo_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
