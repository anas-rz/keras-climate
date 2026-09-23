import os

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from geopandas import GeoDataFrame
from keras import ops
from pyproj import CRS
from rasterio.transform import from_origin

from keras_climate.datasets import (
    DatasetNotFoundError,
    GeoDataset,
    IntersectionDataset,
    NonGeoClassificationDataset,
    NonGeoDataset,
    RasterDataset,
    UnionDataset,
    VectorDataset,
)

MINT = pd.Timestamp(2025, 4, 24)
MAXT = pd.Timestamp(2025, 4, 25)


class CustomGeoDataset(GeoDataset):
    def __init__(
        self, bounds=[(0, 1, 2, 3, MINT, MAXT)], crs=None, res=(1, 1), paths=None
    ):
        if crs is None:
            crs = CRS.from_epsg(4087)
        data = {"filepath": ["file.tif"] * len(bounds)}
        geometry = [shapely.box(b[0], b[2], b[1], b[3]) for b in bounds]
        index = pd.IntervalIndex.from_tuples(
            [(b[4], b[5]) for b in bounds], closed="both", name="datetime"
        )
        self.index = GeoDataFrame(data, index=index, geometry=geometry, crs=crs)
        self.res = res
        self.paths = paths or []

    def __getitem__(self, index):
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        return {"bounds": self._slice_to_tensor(index)}


class CustomNonGeoDataset(NonGeoDataset):
    def __getitem__(self, index):
        return {"index": ops.convert_to_tensor(index)}

    def __len__(self):
        return 2


def _write_tif(path, size=10, count=3, dtype="uint8", value_offset=0):
    transform = from_origin(0, size, 1, 1)
    data = (
        np.arange(count * size * size).reshape(count, size, size) + value_offset
    ) % 255
    data = data.astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def raster_dir(tmp_path):
    d = tmp_path / "raster"
    d.mkdir()
    _write_tif(d / "img_20210101.tif")
    return str(d)


@pytest.fixture
def raster_dataset_cls():
    class MyRaster(RasterDataset):
        filename_glob = "*.tif"
        filename_regex = r"^.*_(?P<date>\d{8})\.tif$"
        is_image = True

    return MyRaster


class TestGeoDataset:
    def test_len(self):
        ds = CustomGeoDataset()
        assert len(ds) == 1

    def test_and(self):
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        result = ds1 & ds2
        assert isinstance(result, IntersectionDataset)

    def test_or(self):
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset(bounds=[(5, 6, 5, 6, MINT, MAXT)])
        result = ds1 | ds2
        assert isinstance(result, UnionDataset)

    def test_add_not_supported(self):
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        with pytest.raises(TypeError):
            ds1 + ds2

    def test_str(self):
        ds = CustomGeoDataset()
        out = str(ds)
        assert "CustomGeoDataset Dataset" in out
        assert "type: GeoDataset" in out

    def test_crs_setter_noop(self, capsys):
        ds = CustomGeoDataset()
        capsys.readouterr()  # discard __init__'s res-setter message
        ds.crs = ds.crs
        assert capsys.readouterr().out == ""

    def test_crs_setter_changes(self, capsys):
        ds = CustomGeoDataset()
        ds.crs = CRS.from_epsg(4326)
        assert "Converting" in capsys.readouterr().out

    def test_res_setter_scalar(self):
        ds = CustomGeoDataset()
        ds.res = 2.0
        assert ds.res == (2.0, 2.0)

    def test_getitem_out_of_bounds_raises(self):
        ds = CustomGeoDataset()
        x, y, t = ds.bounds
        with pytest.raises(IndexError):
            ds[slice(100, 101), y, t]


class TestRasterDataset:
    def test_getitem_shape_and_dtype(self, raster_dataset_cls, raster_dir):
        ds = raster_dataset_cls(paths=raster_dir)
        x, y, t = ds.bounds
        sample = ds[x, y, t]
        assert tuple(sample["image"].shape) == (10, 10, 3)
        assert "bounds" in sample
        assert "transform" in sample

    def test_len(self, raster_dataset_cls, raster_dir):
        ds = raster_dataset_cls(paths=raster_dir)
        assert len(ds) == 1

    def test_not_found_raises(self, raster_dataset_cls, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(DatasetNotFoundError):
            raster_dataset_cls(paths=str(empty))

    def test_mask_dataset_squeezes_channel(self, tmp_path):
        d = tmp_path / "masks"
        d.mkdir()
        _write_tif(d / "mask_20210101.tif", count=1, dtype="uint8")

        class MyMask(RasterDataset):
            filename_glob = "*.tif"
            filename_regex = r"^.*_(?P<date>\d{8})\.tif$"
            is_image = False

        ds = MyMask(paths=str(d))
        x, y, t = ds.bounds
        sample = ds[x, y, t]
        assert "mask" in sample
        assert len(sample["mask"].shape) == 2

    def test_time_series_stacking(self, tmp_path):
        d = tmp_path / "ts"
        d.mkdir()
        _write_tif(d / "img_20210101.tif")
        _write_tif(d / "img_20210102.tif", value_offset=10)

        class MyTS(RasterDataset):
            filename_glob = "*.tif"
            filename_regex = r"^.*_(?P<date>\d{8})\.tif$"
            is_image = True
            mint = pd.Timestamp(2021, 1, 1)
            maxt = pd.Timestamp(2021, 1, 3)

        ds = MyTS(paths=str(d), time_series=True)
        x, y, t = ds.bounds
        sample = ds[x, y, slice(pd.Timestamp(2021, 1, 1), pd.Timestamp(2021, 1, 3))]
        # (T, H, W, C)
        assert len(sample["image"].shape) == 4
        assert sample["image"].shape[0] == 2

    def test_transforms_applied(self, raster_dataset_cls, raster_dir):
        calls = []

        def transform(sample):
            calls.append(True)
            return sample

        ds = raster_dataset_cls(paths=raster_dir, transforms=transform)
        x, y, t = ds.bounds
        ds[x, y, t]
        assert calls == [True]

    def test_files_property(self, raster_dataset_cls, raster_dir):
        ds = raster_dataset_cls(paths=raster_dir)
        assert len(ds.files) == 1
        assert ds.files[0].endswith(".tif")


class TestVectorDataset:
    @pytest.fixture
    def vector_dir(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import box

        d = tmp_path / "vector"
        d.mkdir()
        gdf = gpd.GeoDataFrame(
            {"class": [1, 2]},
            geometry=[box(1, 1, 4, 4), box(5, 5, 8, 8)],
            crs="EPSG:4326",
        )
        gdf.to_file(d / "vector_2021.geojson", driver="GeoJSON")
        return str(d)

    @pytest.fixture
    def vector_dataset_cls(self):
        class MyVector(VectorDataset):
            filename_glob = "*.geojson"
            filename_regex = r"^vector_(?P<date>\d{4})\.geojson$"
            date_format = "%Y"

        return MyVector

    def test_semantic_segmentation(self, vector_dataset_cls, vector_dir):
        ds = vector_dataset_cls(paths=vector_dir, label_name="class", res=1.0)
        x, y, t = ds.bounds
        sample = ds[x, y, t]
        assert "mask" in sample
        assert len(sample["mask"].shape) == 2

    def test_object_detection(self, vector_dataset_cls, vector_dir):
        ds = vector_dataset_cls(
            paths=vector_dir,
            label_name="class",
            res=1.0,
            task="object_detection",
        )
        x, y, t = ds.bounds
        sample = ds[x, y, t]
        assert "bbox_xyxy" in sample
        assert "label" in sample

    def test_instance_segmentation(self, vector_dataset_cls, vector_dir):
        ds = vector_dataset_cls(
            paths=vector_dir,
            label_name="class",
            res=1.0,
            task="instance_segmentation",
        )
        x, y, t = ds.bounds
        sample = ds[x, y, t]
        assert "mask" in sample
        assert "bbox_xyxy" in sample
        assert "label" in sample

    def test_invalid_task_raises(self, vector_dataset_cls, vector_dir):
        with pytest.raises(ValueError, match="Invalid task"):
            vector_dataset_cls(paths=vector_dir, task="bogus")

    def test_get_label_default(self, vector_dataset_cls, vector_dir):
        ds = vector_dataset_cls(paths=vector_dir)
        assert ds.get_label({"class": 5}) == 1


class TestIntersectionDataset:
    def test_getitem_concatenates_image_channels(self, raster_dataset_cls, tmp_path):
        # Regression test: the default concat_samples collate_fn must combine
        # channels-last images along the channel (last) axis, not axis 0.
        d1 = tmp_path / "a"
        d1.mkdir()
        _write_tif(d1 / "img_20210101.tif", count=3)
        d2 = tmp_path / "b"
        d2.mkdir()
        _write_tif(d2 / "img_20210101.tif", count=2)

        ds1 = raster_dataset_cls(paths=str(d1))
        ds2 = raster_dataset_cls(paths=str(d2))
        ds = ds1 & ds2
        x, y, t = ds.bounds
        sample = ds[x, y, t]
        assert tuple(sample["image"].shape) == (10, 10, 5)

    def test_getitem(self):
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        ds = ds1 & ds2
        x, y, t = ds.bounds
        sample = ds[x, y, t]
        assert "bounds" in sample

    def test_no_spatial_intersection_raises(self):
        ds1 = CustomGeoDataset(bounds=[(0, 1, 0, 1, MINT, MAXT)])
        ds2 = CustomGeoDataset(bounds=[(10, 11, 10, 11, MINT, MAXT)])
        with pytest.raises(RuntimeError, match="no spatial intersection"):
            ds1 & ds2

    def test_wrong_type_raises(self):
        ds1 = CustomGeoDataset()
        with pytest.raises(TypeError):
            IntersectionDataset(ds1, CustomNonGeoDataset())

    def test_str(self):
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        ds = ds1 & ds2
        assert "IntersectionDataset" in str(ds)


class TestUnionDataset:
    def test_getitem(self):
        ds1 = CustomGeoDataset(bounds=[(0, 1, 0, 1, MINT, MAXT)])
        ds2 = CustomGeoDataset(bounds=[(10, 11, 10, 11, MINT, MAXT)])
        ds = ds1 | ds2
        x, y, t = ds.bounds
        sample = ds[x, y, t]
        assert "bounds" in sample

    def test_wrong_type_raises(self):
        ds1 = CustomGeoDataset()
        with pytest.raises(TypeError):
            UnionDataset(ds1, CustomNonGeoDataset())

    def test_str(self):
        ds1 = CustomGeoDataset(bounds=[(0, 1, 0, 1, MINT, MAXT)])
        ds2 = CustomGeoDataset(bounds=[(10, 11, 10, 11, MINT, MAXT)])
        ds = ds1 | ds2
        assert "UnionDataset" in str(ds)


class TestNonGeoDataset:
    def test_len(self):
        assert len(CustomNonGeoDataset()) == 2

    def test_str(self):
        out = str(CustomNonGeoDataset())
        assert "type: NonGeoDataset" in out


class TestNonGeoClassificationDataset:
    @pytest.fixture
    def image_folder(self, tmp_path):
        from PIL import Image

        for cls in ("cat", "dog"):
            d = tmp_path / cls
            d.mkdir()
            for i in range(2):
                img = Image.new("RGB", (8, 8), color=(i * 10, 0, 0))
                img.save(d / f"{i}.png")
        return str(tmp_path)

    def test_len_and_classes(self, image_folder):
        ds = NonGeoClassificationDataset(root=image_folder)
        assert len(ds) == 4
        assert ds.classes == ["cat", "dog"]

    def test_getitem(self, image_folder):
        ds = NonGeoClassificationDataset(root=image_folder)
        sample = ds[0]
        assert tuple(sample["image"].shape) == (8, 8, 3)
        assert "label" in sample

    def test_no_class_folders_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            NonGeoClassificationDataset(root=str(empty))
