import csv
import os

import numpy as np
import pytest
import rasterio

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.biomassters import BioMassters

CSV_COLUMNS = [
    "filename",
    "chip_id",
    "satellite",
    "split",
    "month",
    "size",
    "cksum",
    "s3path_us",
    "s3path_eu",
    "s3path_as",
    "corresponding_agbm",
]

MONTHS = ["September", "October", "November"]
SIZE = 8


def _write_tif(path, num_channels, dtype):
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": num_channels,
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_bounds(0, 0, 1, 1, SIZE, SIZE),
        "height": SIZE,
        "width": SIZE,
    }
    if "float" in dtype:
        data = np.random.randn(SIZE, SIZE).astype(dtype)
    else:
        data = np.random.randint(0, 255, (SIZE, SIZE)).astype(dtype)
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(1, num_channels + 1):
            dst.write(data, i)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    sample_ids = ["chip0", "chip1"]
    csv_rows = []

    for split in ("train", "test"):
        os.makedirs(os.path.join(root, f"{split}_features"), exist_ok=True)
        if split == "train":
            os.makedirs(os.path.join(root, "train_agbm"), exist_ok=True)
        for chip_id in sample_ids:
            for sat, num_channels, n_months in (("S1", 4, MONTHS), ("S2", 11, MONTHS)):
                for idx, month in enumerate(n_months):
                    filename = f"{chip_id}_{sat}_{idx:02d}.tif"
                    csv_rows.append(
                        [
                            filename,
                            chip_id,
                            sat,
                            split,
                            month,
                            "0",
                            "0",
                            "path",
                            "path",
                            "path",
                            f"{chip_id}_agbm.tif",
                        ]
                    )
                    _write_tif(
                        os.path.join(root, f"{split}_features", filename),
                        num_channels,
                        "uint16",
                    )
            if split == "train":
                _write_tif(
                    os.path.join(root, "train_agbm", f"{chip_id}_agbm.tif"),
                    1,
                    "float32",
                )

    with open(
        os.path.join(root, BioMassters.metadata_filename), "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(csv_rows)

    return root


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("sensors", [["S1"], ["S2"], ["S1", "S2"]])
@pytest.mark.parametrize("as_time_series", [True, False])
def test_len(prepared_root, split, sensors, as_time_series):
    ds = BioMassters(
        prepared_root, split=split, sensors=sensors, as_time_series=as_time_series
    )
    assert len(ds) > 0


def test_getitem_not_time_series(prepared_root):
    ds = BioMassters(
        prepared_root, split="train", sensors=["S1", "S2"], as_time_series=False
    )
    x = ds[0]
    assert tuple(x["image_S1"].shape) == (SIZE, SIZE, 4)
    assert tuple(x["image_S2"].shape) == (SIZE, SIZE, 11)
    assert tuple(x["label"].shape) == (SIZE, SIZE, 1)


def test_getitem_time_series(prepared_root):
    ds = BioMassters(
        prepared_root, split="train", sensors=["S1"], as_time_series=True
    )
    x = ds[0]
    assert x["image_S1"].shape[0] == len(MONTHS)
    assert x["image_S1"].shape[-1] == 4


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        BioMassters(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = BioMassters(prepared_root, split="train", sensors=["S1", "S2"])
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()

    sample["prediction"] = sample["label"]
    ds.plot(sample)
    plt.close()
    ds.plot(sample, show_titles=False)
    plt.close()
