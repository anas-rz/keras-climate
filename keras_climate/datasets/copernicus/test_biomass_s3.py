import os

import numpy as np
import pytest
import rasterio
from keras import ops
from rasterio.transform import from_origin

from keras_climate.datasets.errors import DatasetNotFoundError
from keras_climate.datasets.copernicus.biomass_s3 import CopernicusBenchBiomassS3

TRANSFORM = from_origin(10, 50, 0.01, 0.01)
N_BANDS = len(CopernicusBenchBiomassS3.all_bands)


def _write_tif(path, bands, size=8, dtype="float32", nodata_corner=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.random.rand(bands, size, size).astype(dtype)
    if nodata_corner:
        data[:, 0, 0] = -np.inf
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=TRANSFORM,
    ) as dst:
        dst.write(data)


def _prepare_root(tmp_path, split="train"):
    root = str(tmp_path)
    directory = os.path.join(root, "biomass_s3")
    os.makedirs(directory, exist_ok=True)

    pid = "tile_0001"
    filename = "S3A_20180425T054022.tif"

    _write_tif(
        os.path.join(directory, "s3_olci", pid, filename),
        bands=N_BANDS,
        nodata_corner=True,
    )
    _write_tif(os.path.join(directory, "biomass", f"{pid}.tif"), bands=1)

    with open(os.path.join(directory, f"static_fnames-{split}.csv"), "w") as f:
        f.write(f"{pid},{filename}\n")

    return root


def test_getitem_static(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchBiomassS3(root=root, split="train", mode="static")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (282, 282, N_BANDS)
    assert tuple(sample["mask"].shape) == (282, 282)
    assert bool(ops.all(ops.isfinite(sample["image"])))


def test_getitem_time_series(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchBiomassS3(root=root, split="train", mode="time-series")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (1, 282, 282, N_BANDS)
    assert tuple(sample["mask"].shape) == (282, 282)


def test_len(tmp_path):
    root = _prepare_root(tmp_path)
    ds = CopernicusBenchBiomassS3(root=root, split="train")
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CopernicusBenchBiomassS3(root=str(tmp_path), download=False)
