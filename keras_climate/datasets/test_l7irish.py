import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.l7irish import L7Irish, L7IrishImage, L7IrishMask


def _write_tif(path, count, value, dtype):
    transform = from_origin(0, 1, 0.01, 0.01)
    data = np.full((count, 8, 8), value, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    _write_tif(
        os.path.join(root, "L71012025_02520150101.TIF"),
        len(L7IrishImage.all_bands),
        50,
        "uint8",
    )
    _write_tif(os.path.join(root, "L7_p012_r025_newmask2015.TIF"), 1, 128, "uint8")
    return root


def test_image_getitem(prepared_root):
    ds = L7IrishImage(paths=prepared_root)
    sample = ds[ds.bounds]
    assert tuple(sample["image"].shape) == (8, 8, len(L7IrishImage.all_bands))


def test_mask_ordinal_mapping(prepared_root):
    ds = L7IrishMask(paths=prepared_root)
    sample = ds[ds.bounds]
    mask = np.asarray(sample["mask"])
    # 128 maps to ordinal class 2
    assert (mask == 2).all()


def test_l7irish_getitem_and_len(prepared_root):
    ds = L7Irish(paths=prepared_root)
    assert len(ds) == 1
    sample = ds[ds.bounds]
    assert "image" in sample
    assert "mask" in sample


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        L7Irish(paths=str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = L7Irish(paths=prepared_root)
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()
