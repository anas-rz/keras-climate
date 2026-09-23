import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.usavars import USAVars


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    uar_dir = os.path.join(root, "uar")
    os.makedirs(uar_dir, exist_ok=True)

    ids = ["id001", "id002", "id003"]
    transform = from_origin(0, 8, 1, 1)
    for i in ids:
        fname = f"tile_{i}.tif"
        data = np.random.randint(0, 255, (4, 8, 8)).astype("uint8")
        with rasterio.open(
            os.path.join(uar_dir, fname),
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=4,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

    for lab in USAVars.label_urls:
        df = pd.DataFrame(
            {
                "ID": ids,
                lab: np.random.rand(len(ids)),
                "lat": np.random.rand(len(ids)) * 10,
                "lon": np.random.rand(len(ids)) * 10,
            }
        ).set_index("ID")
        df.to_csv(os.path.join(root, f"{lab}.csv"))

    for split in ("train", "val", "test"):
        with open(os.path.join(root, f"{split}_split.txt"), "w") as f:
            f.write("\n".join(f"tile_{i}.tif" for i in ids))

    return root


def test_getitem(prepared_root):
    ds = USAVars(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 4)
    assert tuple(sample["labels"].shape) == (3,)
    assert tuple(sample["centroid_lat"].shape) == (1,)
    assert tuple(sample["centroid_lon"].shape) == (1,)


def test_len(prepared_root):
    ds = USAVars(root=prepared_root, split="train", download=False)
    assert len(ds) == 3


def test_invalid_labels():
    with pytest.raises(AssertionError):
        USAVars(labels=("bogus",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        USAVars(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = USAVars(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
