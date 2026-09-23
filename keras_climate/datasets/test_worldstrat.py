import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import rasterio
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.worldstrat import WorldStrat

TRANSFORM = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)
CRS = rasterio.crs.CRS.from_epsg(4326)


def _write_tiff(path, count, size):
    data = np.random.randint(0, 255, (count, size, size), dtype=np.uint16)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=count,
        dtype=np.uint16, transform=TRANSFORM, crs=CRS,
    ) as dst:
        dst.write(data)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    tile = "AOI-1"
    hr_tile_dir = os.path.join(root, "hr_dataset", tile)
    l1c_dir = os.path.join(root, "lr_dataset", tile, "L1C")
    l2a_dir = os.path.join(root, "lr_dataset", tile, "L2A")
    for d in (hr_tile_dir, l1c_dir, l2a_dir):
        os.makedirs(d)

    img_size = 8
    _write_tiff(os.path.join(hr_tile_dir, f"{tile}_ps.tiff"), 4, img_size)
    _write_tiff(os.path.join(hr_tile_dir, f"{tile}_pan.tiff"), 1, img_size)
    _write_tiff(os.path.join(hr_tile_dir, f"{tile}_rgbn.tiff"), 4, img_size // 4)

    rgbn = np.random.randint(0, 255, (img_size, img_size, 4), dtype=np.uint8)
    Image.fromarray(rgbn, mode="RGBA").save(os.path.join(hr_tile_dir, f"{tile}_rgb.png"))

    base_date = date(2021, 1, 1)
    dates = [base_date + timedelta(days=i * 30) for i in range(4)]
    write_order = [3, 1, 4, 2]
    for n in write_order:
        _write_tiff(os.path.join(l1c_dir, f"{tile}-{n}-L1C_data.tiff"), 13, img_size // 2)
        _write_tiff(os.path.join(l2a_dir, f"{tile}-{n}-L2A_data.tiff"), 12, img_size // 2)

    metadata = [
        {
            "tile": tile,
            "n": n,
            "lon": 10.0,
            "lat": 20.0,
            "lowres_date": dates[n - 1].strftime("%Y-%m-%d"),
            "highres_date": dates[0].strftime("%Y-%m-%d"),
        }
        for n in write_order
    ]
    pd.DataFrame(metadata).to_csv(os.path.join(root, "metadata.csv"), index=False)
    pd.DataFrame([{"tile": tile, "split": "train"}]).to_csv(
        os.path.join(root, "stratified_train_val_test_split.csv"), index=False
    )

    return root


def test_getitem(prepared_root):
    ds = WorldStrat(root=prepared_root, split="train", download=False)
    sample = ds[0]
    for modality in ds.modalities:
        assert sample[f"image_{modality}"].dtype == "float32"

    low_res_date = sample["low_res_date"]
    assert tuple(low_res_date.shape) == (4,)
    assert tuple(sample["image_l1c"].shape)[0] == 4


def test_sentinel_paths_sorted_by_index(prepared_root):
    ds = WorldStrat(root=prepared_root, split="train", download=False)
    aoi = ds.file_path_df["tile"][0]
    data_dir = os.path.join(ds.root, ds.lr_dir, aoi, "L1C")
    pairs = ds._sentinel_paths(data_dir)
    assert [n for n, _ in pairs] == [1, 2, 3, 4]


def test_len(prepared_root):
    ds = WorldStrat(root=prepared_root, split="train", download=False)
    assert len(ds) == 1
    ds_val = WorldStrat(root=prepared_root, split="val", download=False)
    assert len(ds_val) == 0


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        WorldStrat(root=prepared_root, split="bogus")


def test_invalid_modality(prepared_root):
    with pytest.raises(AssertionError):
        WorldStrat(root=prepared_root, modalities=("not_a_modality",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        WorldStrat(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = WorldStrat(root=prepared_root, split="train", download=False)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()


def test_plot_prediction(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = WorldStrat(root=prepared_root, split="train", download=False)
    sample = ds[0]
    sample["prediction"] = sample["image_hr_rgbn"]
    ds.plot(sample)
    plt.close()
