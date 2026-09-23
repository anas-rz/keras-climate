import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.idtrees import IDTReeS

laspy = pytest.importorskip("laspy")


def _write_tif(path, count, size=4):
    transform = from_origin(0, 10, 10 / size, 10 / size)
    data = np.random.rand(count, size, size).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=count,
        dtype="float32",
        crs="EPSG:32617",
        transform=transform,
    ) as dst:
        dst.write(data)


def _write_las(path):
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([1.0, 2.0, 3.0])
    las.y = np.array([1.0, 2.0, 3.0])
    las.z = np.array([1.0, 2.0, 3.0])
    las.write(path)


def _write_remote_sensing(directory, fname):
    for sub, count in [("RGB", 3), ("HSI", 4), ("CHM", 1)]:
        d = os.path.join(directory, "RemoteSensing", sub)
        os.makedirs(d, exist_ok=True)
        _write_tif(os.path.join(d, fname), count)
    las_dir = os.path.join(directory, "RemoteSensing", "LAS")
    os.makedirs(las_dir, exist_ok=True)
    _write_las(os.path.join(las_dir, fname.replace(".tif", ".las")))


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)

    # --- train split ---
    train_dir = os.path.join(root, "train")
    _write_remote_sensing(train_dir, "SITE_1.tif")

    itc_dir = os.path.join(train_dir, "ITC")
    os.makedirs(itc_dir, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"id": [1], "indvdID": ["IND001"], "geometry": [box(2, 2, 4, 4)]}
    )
    gdf.to_file(os.path.join(itc_dir, "train_SITE.shp"))

    field_dir = os.path.join(train_dir, "Field")
    os.makedirs(field_dir, exist_ok=True)
    with open(os.path.join(field_dir, "itc_rsFile.csv"), "w") as f:
        f.write("id,indvdID,rsFile\n")
        f.write("1,IND001,SITE_1.tif\n")
    with open(os.path.join(field_dir, "train_data.csv"), "w") as f:
        f.write("indvdID,taxonID\n")
        f.write("IND001,ACPE\n")

    # --- test split, task1 ---
    task1_dir = os.path.join(root, "task1")
    _write_remote_sensing(task1_dir, "SITE_1.tif")

    # --- test split, task2 ---
    task2_dir = os.path.join(root, "task2")
    _write_remote_sensing(task2_dir, "SITE_1.tif")
    itc2_dir = os.path.join(task2_dir, "ITC")
    os.makedirs(itc2_dir, exist_ok=True)
    gdf2 = gpd.GeoDataFrame(
        {
            "indvdID": ["IND002"],
            "plotID": ["SITE_1.tif"],
            "geometry": [box(2, 2, 4, 4)],
        }
    )
    gdf2.to_file(os.path.join(itc2_dir, "test_SITE.shp"))

    return root


def test_train_getitem(prepared_root):
    ds = IDTReeS(root=prepared_root, split="train")
    assert len(ds) == 1
    sample = ds[0]
    assert tuple(sample["image"].shape) == (200, 200, 3)
    assert tuple(sample["hsi"].shape) == (200, 200, 4)
    assert tuple(sample["chm"].shape) == (200, 200, 1)
    assert "bbox_xyxy" in sample
    assert "label" in sample
    assert sample["las"].shape[0] == 3


def test_test_task1_getitem(prepared_root):
    ds = IDTReeS(root=prepared_root, split="test", task="task1")
    assert len(ds) == 1
    sample = ds[0]
    assert "bbox_xyxy" not in sample


def test_test_task2_getitem(prepared_root):
    ds = IDTReeS(root=prepared_root, split="test", task="task2")
    assert len(ds) == 1
    sample = ds[0]
    assert "bbox_xyxy" in sample


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        IDTReeS(root=prepared_root, split="bogus")


def test_invalid_task(prepared_root):
    with pytest.raises(AssertionError):
        IDTReeS(root=prepared_root, split="test", task="bogus")


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        IDTReeS(root=str(tmp_path), split="train")


def test_filter_boxes_removes_degenerate():
    from keras import ops

    ds = object.__new__(IDTReeS)
    boxes = ops.convert_to_tensor(
        np.array([[0, 0, 5, 5], [1, 1, 1, 1]], dtype="float32")
    )
    labels = ops.convert_to_tensor(np.array([0, 1], dtype="int64"))
    filtered_boxes, filtered_labels = ds._filter_boxes(
        image_size=(10, 10), min_size=1, boxes=boxes, labels=labels
    )
    assert tuple(filtered_boxes.shape) == (1, 4)
    assert tuple(filtered_labels.shape) == (1,)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = IDTReeS(root=prepared_root, split="train")
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()
