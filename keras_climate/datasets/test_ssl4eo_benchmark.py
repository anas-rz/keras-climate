import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.ssl4eo_benchmark import SSL4EOLBenchmark


def _write_tif(path, bands, size=4, dtype="float32", data=None):
    transform = from_origin(0, size, 1, 1)
    if data is None:
        data = np.random.rand(bands, size, size).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    sensor = "oli_sr"
    product = "cdl"
    img_dir_name = SSL4EOLBenchmark.image_root.format(sensor)
    mask_dir_name = SSL4EOLBenchmark.mask_dir_dict[sensor].format(product)
    year = SSL4EOLBenchmark.year_dict[sensor]

    for i in range(10):
        scene = f"scene_{i:03}"
        img_dir = os.path.join(root, img_dir_name, scene)
        mask_dir = os.path.join(root, mask_dir_name, scene)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        _write_tif(os.path.join(img_dir, "all_bands.tif"), 7)
        mask_data = np.random.randint(0, 50, (1, 4, 4)).astype("uint8")
        _write_tif(
            os.path.join(mask_dir, f"{product}_{year}.tif"),
            1,
            dtype="uint8",
            data=mask_data,
        )

    return root


def test_getitem(prepared_root):
    ds = SSL4EOLBenchmark(
        root=prepared_root, sensor="oli_sr", product="cdl", split="train"
    )
    sample = ds[0]
    assert tuple(sample["image"].shape) == (4, 4, 7)
    assert tuple(sample["mask"].shape) == (4, 4)


def test_len(prepared_root):
    ds_train = SSL4EOLBenchmark(
        root=prepared_root, sensor="oli_sr", product="cdl", split="train"
    )
    ds_val = SSL4EOLBenchmark(
        root=prepared_root, sensor="oli_sr", product="cdl", split="val"
    )
    ds_test = SSL4EOLBenchmark(
        root=prepared_root, sensor="oli_sr", product="cdl", split="test"
    )
    assert len(ds_train) + len(ds_val) + len(ds_test) <= 10
    assert len(ds_train) == 7


def test_invalid_sensor():
    with pytest.raises(AssertionError):
        SSL4EOLBenchmark(sensor="bogus")


def test_invalid_product():
    with pytest.raises(AssertionError):
        SSL4EOLBenchmark(product="bogus")


def test_invalid_split():
    with pytest.raises(AssertionError):
        SSL4EOLBenchmark(split="bogus")


def test_invalid_classes():
    with pytest.raises(AssertionError):
        SSL4EOLBenchmark(product="cdl", classes=[99999])


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SSL4EOLBenchmark(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SSL4EOLBenchmark(
        root=prepared_root, sensor="oli_sr", product="cdl", split="train"
    )
    ds.plot(ds[0], suptitle="Test")
    plt.close()
