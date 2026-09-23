import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, RGBBandsMissingError
from keras_climate.datasets.eurosat import EuroSAT, EuroSAT100, EuroSATSpatial


@pytest.fixture(params=[EuroSAT, EuroSATSpatial, EuroSAT100])
def dataset_cls(request):
    return request.param


@pytest.fixture
def prepared_root(tmp_path, dataset_cls):
    # split_filenames differ per subclass; write the files under each subclass' names
    base = os.path.join(str(tmp_path), dataset_cls.base_dir)
    os.makedirs(base, exist_ok=True)
    classes = ("AnnualCrop", "Forest")
    transform = from_origin(0, 8, 1, 1)
    all_files = []
    for cls in classes:
        class_dir = os.path.join(base, cls)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(2):
            fname = f"{cls}_{i}.tif"
            path = os.path.join(class_dir, fname)
            data = np.random.randint(0, 255, (13, 8, 8)).astype("uint16")
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=8,
                width=8,
                count=13,
                dtype="uint16",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data)
            all_files.append(fname)

    half = len(all_files) // 2
    splits = {"train": all_files[:half], "val": all_files[:half], "test": all_files[half:]}
    for split, fnames in splits.items():
        with open(
            os.path.join(str(tmp_path), dataset_cls.split_filenames[split]), "w"
        ) as f:
            f.write("\n".join(fnames) + "\n")

    return str(tmp_path)


def test_getitem(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 13)
    assert "label" in sample


def test_len(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, split="train", download=False)
    assert len(ds) == 2


def test_rgb_band_subset(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, split="train", bands=("B04", "B03", "B02"))
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)


def test_invalid_bands():
    with pytest.raises(ValueError):
        EuroSAT(bands=("OK", "BK"))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        EuroSAT(root=str(tmp_path), download=False)


def test_plot_rgb_missing_band_raises(dataset_cls, prepared_root):
    ds = dataset_cls(root=prepared_root, split="train", bands=("B03",))
    with pytest.raises(RGBBandsMissingError):
        ds.plot(ds[0])


def test_plot(dataset_cls, prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = dataset_cls(root=prepared_root, split="train", bands=("B04", "B03", "B02"))
    ds.plot(ds[0], suptitle="Test")
    plt.close()
