import os

import numpy as np
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.sustainbench_crop_yield import SustainBenchCropYield


@pytest.fixture
def prepared_root(tmp_path):
    country_dir = os.path.join(str(tmp_path), "soybeans", "usa")
    os.makedirs(country_dir, exist_ok=True)

    for split in ("train", "dev", "test"):
        n = 3
        images = np.random.rand(n, 32, 32, 9).astype("float32")
        yields = np.random.rand(n).astype("float32")
        years = np.random.randint(2000, 2020, size=(n,)).astype("int64")
        ndvi = np.random.rand(n, 32).astype("float32")

        np.savez(os.path.join(country_dir, f"{split}_hists.npz"), data=images)
        np.savez(os.path.join(country_dir, f"{split}_yields.npz"), data=yields)
        np.savez(os.path.join(country_dir, f"{split}_years.npz"), data=years)
        np.savez(os.path.join(country_dir, f"{split}_ndvi.npz"), data=ndvi)

    return str(tmp_path)


def test_getitem(prepared_root):
    ds = SustainBenchCropYield(root=prepared_root, split="train", countries=["usa"])
    sample = ds[0]
    assert tuple(sample["image"].shape) == (32, 32, 9)
    assert "label" in sample
    assert "year" in sample
    assert "ndvi" in sample


def test_len(prepared_root):
    ds = SustainBenchCropYield(root=prepared_root, split="train", countries=["usa"])
    assert len(ds) == 3


def test_invalid_split():
    with pytest.raises(AssertionError):
        SustainBenchCropYield(split="bogus")


def test_invalid_country():
    with pytest.raises(AssertionError):
        SustainBenchCropYield(countries=["bogus"])


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SustainBenchCropYield(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SustainBenchCropYield(root=prepared_root, split="train", countries=["usa"])
    ds.plot(ds[0], suptitle="Test")
    plt.close()
