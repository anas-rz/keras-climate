"""Base classes for keras_climate datasets (ported from torchgeo.datasets.geo).

This is a multi-backend Keras port of torchgeo's dataset framework. Sample
values are Keras tensors (via ``keras.ops.convert_to_tensor``) instead of
``torch.Tensor``, so datasets work unchanged under the TensorFlow, JAX, and
PyTorch Keras 3 backends. Raster/vector I/O (rasterio, geopandas, pyproj,
shapely) is unaffected by the tensor backend and is required regardless.

.. important::
   Unlike torchgeo (channels-first ``C x H x W``), image/mask samples here
   use the channels-last layout (``H x W x C``) to match the rest of
   keras_climate and Keras's own default ``Conv2D`` data format. This is a
   deliberate deviation from the upstream API, applied consistently across
   :class:`RasterDataset`, :class:`VectorDataset`, and
   :class:`NonGeoClassificationDataset`.
"""

import abc
import functools
import os
import re
import warnings
from contextlib import ExitStack
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pathlib
import pyproj
import rasterio
import rasterio.features
import rasterio.merge
import shapely
from geopandas import GeoDataFrame
from keras import ops
from matplotlib.colors import ListedColormap
from pyproj import CRS as PROJ_CRS
from rasterio.crs import CRS as RIO_CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds, from_gcps
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from shapely import MultiPolygon, Polygon

from .errors import DatasetNotFoundError
from .mixins import PlottingMixin
from .utils import (
    array_to_tensor,
    concat_samples,
    convert_poly_coords,
    disambiguate_timestamp,
    find_files,
    lazy_import,
    merge_samples,
)

_FLOATING_DTYPES = {"float16", "bfloat16", "float32", "float64"}

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def _pil_loader(path):
    Image = lazy_import("PIL.Image")
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def _find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def _make_dataset(directory, class_to_idx, is_valid_file=None):
    directory = os.path.expanduser(directory)
    if is_valid_file is None:

        def is_valid_file(path):
            return str(path).lower().endswith(IMG_EXTENSIONS)

    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    instances.append((path, class_index))
    return instances


class GeoDataset(abc.ABC, PlottingMixin):
    """Abstract base class for datasets containing geospatial information.

    Geospatial information includes things like coordinates, CRS, and
    resolution. Unlike :class:`NonGeoDataset`, the presence of geospatial
    information allows two or more datasets to be combined based on
    latitude/longitude, via ``dataset1 & dataset2`` (:class:`IntersectionDataset`)
    or ``dataset1 | dataset2`` (:class:`UnionDataset`).
    """

    index: GeoDataFrame
    paths = "data"
    _res = (0.0, 0.0)

    #: Glob expression used to search for files.
    filename_glob = "*"

    #: GeoDataset addition can be ambiguous and is no longer supported.
    #: Use the intersection or union operator instead.
    __add__ = None

    def _disambiguate_slice(self, index):
        out = list(self.bounds)

        if isinstance(index, slice):
            index = (index,)

        for i in range(len(index)):
            if index[i].start is not None:
                out[i] = slice(index[i].start, out[i].stop, out[i].step)
            if index[i].stop is not None:
                out[i] = slice(out[i].start, index[i].stop, out[i].step)
            if index[i].step is not None:
                out[i] = slice(out[i].start, out[i].stop, index[i].step)

        geoslice = tuple(out)
        assert len(geoslice) == 3
        return geoslice

    def _slice_to_tensor(self, index):
        x, y, t = self._disambiguate_slice(index)
        bounds = [
            x.start,
            x.stop,
            x.step,
            y.start,
            y.stop,
            y.step,
            t.start.timestamp(),
            t.stop.timestamp(),
            t.step,
        ]
        return ops.convert_to_tensor(np.array(bounds, dtype="float64"))

    @abc.abstractmethod
    def __getitem__(self, index):
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice."""

    def __and__(self, other):
        return IntersectionDataset(self, other)

    def __or__(self, other):
        return UnionDataset(self, other)

    def __len__(self):
        return len(self.index)

    def __str__(self):
        return f"""\
{self.__class__.__name__} Dataset
    type: GeoDataset
    bbox: {self.bounds}
    size: {len(self)}"""

    @property
    def bounds(self):
        xmin, ymin, xmax, ymax = self.index.total_bounds
        xres, yres = self.res
        tmin = self.index.index.left.min()
        tmax = self.index.index.right.max()
        tres = 1
        return slice(xmin, xmax, xres), slice(ymin, ymax, yres), slice(tmin, tmax, tres)

    @property
    def crs(self):
        return self.index.crs

    @crs.setter
    def crs(self, new_crs):
        if new_crs == self.crs:
            return

        print(f"Converting {self.__class__.__name__} CRS from {self.crs} to {new_crs}")
        self.index.to_crs(new_crs, inplace=True)

    @property
    def res(self):
        return self._res

    @res.setter
    def res(self, new_res):
        if isinstance(new_res, (int, float)):
            new_res = (new_res, new_res)

        if new_res == self.res:
            return

        print(f"Converting {self.__class__.__name__} res from {self.res} to {new_res}")
        self._res = new_res

    @property
    def files(self):
        if isinstance(self.paths, (str, os.PathLike)):
            paths = [self.paths]
        else:
            paths = self.paths

        # Using set to remove any duplicates if directories are overlapping
        files = set()
        for path in paths:
            found = set(find_files(path, self.filename_glob))
            if found:
                files.update(found)
            elif not os.path.isdir(path) and not hasattr(self, "download"):
                warnings.warn(
                    f"Could not find any relevant files for provided path '{path}'. "
                    f"Path was ignored.",
                    UserWarning,
                )
        return sorted(files)


class RasterDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files."""

    #: Regular expression used to extract date from filename.
    filename_regex = ".*"

    #: Date format string used to parse date from filename.
    date_format = "%Y%m%d"

    #: Minimum timestamp if not in filename
    mint = pd.Timestamp.min

    #: Maximum timestamp if not in filename
    maxt = pd.Timestamp.max

    #: True if the dataset only contains model inputs (images). False if the
    #: dataset only contains ground truth model outputs (masks).
    is_image = True

    #: True if data is stored in a separate file for each band, else False.
    separate_files = False

    #: Nodata value for the dataset. If None, the source files' nodata value is used.
    nodata = None

    @property
    def dtype(self):
        return "float32" if self.is_image else "int64"

    @property
    def resampling(self):
        return Resampling.bilinear if self.dtype in _FLOATING_DTYPES else Resampling.nearest

    def __init__(
        self,
        paths="data",
        crs=None,
        res=None,
        bands=None,
        transforms=None,
        cache=True,
        time_series=False,
    ):
        """Initialize a new RasterDataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS (defaults to the
                resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            time_series: if True, stack data along the time series dimension
                (``[T, H, W, C]``). If False, merge data into a mosaic
                (``[H, W, C]``). For mask-style datasets (``is_image=False``),
                single-band data has the channel dimension squeezed.
        """
        self.paths = paths
        self.bands = bands or self.all_bands
        self.transforms = transforms
        self.cache = cache
        self.time_series = time_series

        if self.all_bands:
            assert set(self.bands) <= set(self.all_bands)

        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        filepaths = []
        datetimes = []
        geometries = []
        for filepath in self.files:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                vrt = None
                try:
                    vrt = self._load_warp_file(filepath=filepath, crs=crs)
                    if self.cmap is None:
                        try:
                            colors = np.array(list(vrt.colormap(1).values())) / 255
                            self.cmap = ListedColormap(colors)
                        except ValueError:
                            pass
                    if crs is None:
                        with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
                            crs = PROJ_CRS.from_user_input(vrt.crs)
                    footprint = self.footprint_from_datasource(vrt)
                    if footprint is None:
                        footprint = shapely.box(*vrt.bounds)
                    geometries.append(footprint)
                    if res is None:
                        res = vrt.res
                except rasterio.errors.RasterioIOError:
                    continue
                else:
                    filepaths.append(filepath)
                    mint, maxt = self._filepath_to_timestamp(filepath)
                    datetimes.append((mint, maxt))
                finally:
                    if vrt is not None:
                        vrt.close()

        if len(filepaths) == 0:
            raise DatasetNotFoundError(self)

        if not self.separate_files:
            self.band_indexes = None
            if self.bands:
                if self.all_bands:
                    self.band_indexes = [
                        self.all_bands.index(i) + 1 for i in self.bands
                    ]
                else:
                    msg = (
                        f"{self.__class__.__name__} is missing an `all_bands` "
                        "attribute, so `bands` cannot be specified."
                    )
                    raise AssertionError(msg)

        if res is not None:
            if isinstance(res, (int, float)):
                res = (res, res)

            self._res = res

        data = {"filepath": filepaths}
        index = pd.IntervalIndex.from_tuples(datetimes, closed="both", name="datetime")
        self.index = GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    def __getitem__(self, index):
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        out_crs = self.crs

        if self.separate_files:
            data_list = []
            for band in self.bands:
                band_filepaths = []
                for filepath in df.filepath:
                    filepath = self._update_filepath(band, filepath)
                    band_filepaths.append(filepath)
                data_list.append(
                    self._merge_or_stack(band_filepaths, index, out_crs=out_crs)
                )
            data = ops.concatenate(data_list, axis=-1)
        else:
            data = self._merge_or_stack(
                df.filepath, index, self.band_indexes, out_crs=out_crs
            )

        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            "bounds": self._slice_to_tensor(index),
            "transform": ops.convert_to_tensor(np.array(list(transform))),
        }

        data = ops.cast(data, self.dtype)
        if self.is_image:
            sample["image"] = data
        else:
            # Only squeeze a genuinely single-band mask; multi-band masks
            # (e.g. separate_files=True with >1 band) are left as (H, W, C).
            sample["mask"] = ops.squeeze(data, axis=-1) if data.shape[-1] == 1 else data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _filepath_to_timestamp(self, filepath):
        mint = self.mint
        maxt = self.maxt

        filename = os.path.basename(filepath)
        match = re.match(self.filename_regex, filename, re.VERBOSE)
        if match:
            if "date" in match.groupdict():
                date = match.group("date")
                mint, maxt = disambiguate_timestamp(date, self.date_format)
            elif "start" in match.groupdict() and "stop" in match.groupdict():
                start = match.group("start")
                stop = match.group("stop")
                mint, _ = disambiguate_timestamp(start, self.date_format)
                _, maxt = disambiguate_timestamp(stop, self.date_format)

        return mint, maxt

    def _update_filepath(self, band, filepath):
        filename = os.path.basename(filepath)
        directory = os.path.dirname(filepath)
        match = re.match(self.filename_regex, filename, re.VERBOSE)
        if match and "band" in match.groupdict():
            start = match.start("band")
            end = match.end("band")
            filename = filename[:start] + band + filename[end:]
        return os.path.join(directory, filename)

    def _merge_or_stack(self, filepaths, index, band_indexes=None, out_crs=None):
        """Load and combine one or more files.

        If *time_series* is True, files are stacked into a ``[T, H, W, C]``
        shape. If *time_series* is False, files are merged into a
        ``[H, W, C]`` mosaic.
        """
        out_crs = out_crs or self.crs
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp, out_crs) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp, out_crs) for fp in filepaths]

        x, y, _ = self._disambiguate_slice(index)
        kwargs = {
            "bounds": (x.start, y.start, x.stop, y.stop),
            "res": (x.step, y.step),
            "indexes": band_indexes,
            "resampling": self.resampling,
        }

        if self.time_series:
            dest = np.stack(
                [rasterio.merge.merge([fh], **kwargs)[0] for fh in vrt_fhs]
            )
            dest = np.transpose(dest, (0, 2, 3, 1))
        else:
            dest = rasterio.merge.merge(vrt_fhs, **kwargs)[0]
            dest = np.transpose(dest, (1, 2, 0))

        return array_to_tensor(dest)

    @functools.lru_cache(maxsize=128)  # noqa: B019
    def _cached_load_warp_file(self, filepath, crs):
        return self._load_warp_file(filepath, crs)

    def _load_warp_file(self, filepath, crs=None):
        src = rasterio.open(filepath)

        has_meaningful_affine = (
            src.transform is not None and not src.transform.is_identity
        )
        if has_meaningful_affine:
            src_crs, src_transform = src.crs, src.transform
        else:
            try:
                src_crs, src_transform = self._compute_affine_georeferencing(src)
            except ValueError:
                src.close()
                raise

        try:
            dst_crs = RIO_CRS.from_user_input(crs or self.crs)
        except AttributeError:
            dst_crs = src_crs

        dst_transform, dst_width, dst_height, needs_warp = (
            self._compute_affine_warp_grid(
                src_crs, src_transform, src.width, src.height, dst_crs
            )
        )

        if needs_warp or self.nodata is not None:
            override = {}
            if self.nodata is not None:
                override["src_nodata"] = self.nodata
            vrt = WarpedVRT(
                src,
                crs=dst_crs,
                transform=dst_transform,
                height=dst_height,
                width=dst_width,
                src_crs=src_crs,
                src_transform=src_transform,
                **override,
            )
            src.close()
            return vrt
        return src

    def _compute_affine_georeferencing(self, src):
        gcps, gcp_crs = src.gcps
        if not gcps or gcp_crs is None:
            raise ValueError(
                f"{src.name}: dataset has no usable affine CRS/transform and no GCP CRS."
            )

        return gcp_crs, from_gcps(gcps)

    def _compute_affine_warp_grid(self, crs, transform, width, height, dst_crs):
        west, south, east, north = array_bounds(height, width, transform)

        left = min(west, east)
        bottom = min(south, north)
        right = max(west, east)
        top = max(south, north)

        dst_transform, dst_width, dst_height = calculate_default_transform(
            crs, dst_crs, width, height, left, bottom, right, top
        )

        needs_warp = (
            (crs != dst_crs)
            or (not transform.almost_equals(dst_transform))
            or (width != dst_width)
            or (height != dst_height)
        )

        return dst_transform, dst_width, dst_height, needs_warp

    def footprint_from_datasource(self, datasource):
        """Compute the spatial footprint of the dataset from a file handle.

        Override this in subclasses to compute a more precise footprint than
        just the raster bounds.
        """
        return


class XarrayDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files (xarray).

    .. warning::
       This dataset is considered experimental and subject to change.
    """

    #: Nodata value for the dataset. If None, the source files' nodata value is used.
    nodata = None

    def __init__(self, paths="data", crs=None, res=None, data_vars=None, transforms=None):
        lazy_import("rioxarray")
        xr = lazy_import("xarray")
        self.paths = paths
        self.transforms = transforms

        filepaths = []
        datetimes = []
        geometries = []
        for filepath in self.files:
            try:
                with xr.open_dataset(filepath, decode_coords="all") as src:
                    crs = crs or src.rio.crs or PROJ_CRS.from_epsg(4326)
                    res = res or src.rio.resolution()
                    data_vars = data_vars or list(src.data_vars.keys())
                    tmin = pd.Timestamp(src.time.values.min())
                    tmax = pd.Timestamp(src.time.values.max())

                    if src.rio.crs is None:
                        warnings.warn(
                            f"Unable to decode coordinates of '{filepath}', "
                            f"defaulting to {crs}. Set `crs` if this is incorrect.",
                            UserWarning,
                        )
                        src = src.rio.write_crs(crs)

                    if src.rio.crs != crs:
                        src = src.rio.reproject(crs)

                    filepaths.append(filepath)
                    datetimes.append((tmin, tmax))
                    geometries.append(shapely.box(*src.rio.bounds()))
            except (OSError, ValueError):
                continue

        if len(filepaths) == 0:
            raise DatasetNotFoundError(self)

        if res is not None:
            if isinstance(res, (int, float)):
                res = (res, res)

            self._res = res

        if data_vars is not None:
            self.data_vars = data_vars

        data = {"filepath": filepaths}
        index = pd.IntervalIndex.from_tuples(datetimes, closed="both", name="datetime")
        self.index = GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    def __getitem__(self, index):
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        out_crs = self.crs

        image = self._merge_files(df.filepath, index, out_crs=out_crs)
        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            "bounds": self._slice_to_tensor(index),
            "image": image,
            "transform": ops.convert_to_tensor(np.array(list(transform))),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(self, filepaths, index, out_crs):
        xr = lazy_import("xarray")
        rioxr = lazy_import("rioxarray")
        lazy_import("rioxarray.merge")

        x, y, t = self._disambiguate_slice(index)
        bounds = (x.start, y.start, x.stop, y.stop)
        res = (abs(x.step), abs(y.step))

        with ExitStack() as stack:
            datasets = []
            for filepath in filepaths:
                src = stack.enter_context(
                    xr.open_dataset(filepath, decode_times=True, decode_coords="all")
                )

                if src.rio.crs is None:
                    src = src.rio.write_crs(self.crs)

                y_dim = src.rio.y_dim
                if src[y_dim][0] < src[y_dim][-1]:
                    src = src.isel({y_dim: slice(None, None, -1)})

                if src.rio.crs != out_crs or res != src.rio.resolution():
                    src = src.rio.reproject(out_crs, resolution=res)

                if self.nodata is not None:
                    for var in self.data_vars:
                        src[var] = src[var].rio.write_nodata(self.nodata)

                datasets.append(src)

            dataset = rioxr.merge.merge_datasets(
                datasets, bounds=bounds, res=res, nodata=self.nodata, crs=out_crs
            )
            dataset = dataset.sel(time=t)

            tensors = []
            for var in self.data_vars:
                tensors.append(array_to_tensor(dataset[var].values))

        # (C, H, W) -> (H, W, C)
        return ops.transpose(ops.stack(tensors), (1, 2, 0))


class VectorDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as vector files."""

    #: Regular expression used to extract date from filename.
    filename_regex = ".*"

    #: Date format string used to parse date from filename.
    date_format = "%Y%m%d"

    @property
    def dtype(self):
        return "int64"

    def __init__(
        self,
        paths="data",
        crs=None,
        res=(0.0001, 0.0001),
        transforms=None,
        label_name=None,
        task="semantic_segmentation",
        layer=None,
    ):
        """Initialize a new VectorDataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            label_name: name of the dataset property that has the label to
                be rasterized into the mask
            task: one of 'object_detection', 'semantic_segmentation',
                'instance_segmentation'
            layer: if the input is a multilayer vector dataset, such as a
                geopackage, specify which layer to use
        """
        self.paths = paths
        self.transforms = transforms
        self.label_name = label_name
        allowed_tasks = [
            "semantic_segmentation",
            "object_detection",
            "instance_segmentation",
        ]
        if task not in allowed_tasks:
            raise ValueError(f"Invalid task: {task!r}. Must be one of {allowed_tasks}")
        self.task = task
        self.layer = layer
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        filepaths = []
        datetimes = []
        geometries = []
        for filepath in self.files:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    if pathlib.Path(filepath).suffix.lower() == ".parquet":
                        src = gpd.read_parquet(filepath)
                    else:
                        src = gpd.read_file(filepath, layer=layer)
                    crs = crs or src.crs or PROJ_CRS.from_epsg(4326)
                    if src.crs is None:
                        src.set_crs(crs, inplace=True)
                    elif src.crs != crs:
                        src.to_crs(crs, inplace=True)

                    minx, miny, maxx, maxy = src.total_bounds
                    geom = shapely.box(minx, miny, maxx, maxy)
                    geometries.append(geom)
                except (RuntimeError, ValueError):
                    continue
                else:
                    filepaths.append(filepath)

                    mint = pd.Timestamp.min
                    maxt = pd.Timestamp.max
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    datetimes.append((mint, maxt))

        if len(filepaths) == 0:
            raise DatasetNotFoundError(self)

        if isinstance(res, (int, float)):
            res = (res, res)

        self._res = res

        data = {"filepath": filepaths}
        index = pd.IntervalIndex.from_tuples(datetimes, closed="both", name="datetime")
        self.index = GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    def __getitem__(self, index):
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        out_crs = self.crs

        shapes = []
        for filepath in df.filepath:
            if pathlib.Path(filepath).suffix.lower() == ".parquet":
                src = gpd.read_parquet(filepath)
            else:
                src = gpd.read_file(filepath, layer=self.layer)

            transformer = pyproj.Transformer.from_crs(out_crs, src.crs, always_xy=True)
            (minx, miny) = transformer.transform(x.start, y.start)
            (maxx, maxy) = transformer.transform(x.stop, y.stop)

            src = src.cx[minx:maxx, miny:maxy]
            src.to_crs(out_crs, inplace=True)

            labels = np.array(
                [self.get_label(row) for _, row in src.iterrows()]
            ).astype(np.int32)

            shapes.extend(list(zip(src.geometry, labels)))

        width = (x.stop - x.start) / x.step
        height = (y.stop - y.start) / y.step
        transform = rasterio.transform.from_bounds(
            x.start, y.start, x.stop, y.stop, width, height
        )
        if shapes:
            if self.task == "semantic_segmentation":
                masks = rasterio.features.rasterize(
                    shapes, out_shape=(round(height), round(width)), transform=transform
                )

            elif self.task == "object_detection":
                label_list = []
                box_list = []
                for s in shapes:
                    shape = shapely.geometry.shape(s[0])
                    p = convert_poly_coords(shape, transform, inverse=True)
                    p = shapely.clip_by_rect(p, 0, 0, width, height)

                    label_list.append(s[1])
                    box_list.append(p.bounds)

                labels = np.array(label_list).astype(np.int32)
                boxes_xyxy = np.array(box_list).astype(np.float32)

            elif self.task == "instance_segmentation":
                label_list = []
                box_list = []
                mask_list = []
                for i, s in enumerate(shapes):
                    shape = shapely.geometry.shape(s[0])
                    p = convert_poly_coords(shape, transform, inverse=True)
                    p = shapely.clip_by_rect(p, 0, 0, width, height)

                    label_list.append(s[1])
                    box_list.append(p.bounds)

                    mask = rasterio.features.rasterize(
                        [(s[0], i + 1)],
                        out_shape=(round(height), round(width)),
                        transform=transform,
                    )
                    mask_list.append(mask)

                labels = np.array(label_list).astype(np.int32)
                boxes_xyxy = np.array(box_list).astype(np.float32)
                masks = np.array(mask_list)

                obj_ids = np.unique(masks)
                obj_ids = obj_ids[1:]

                masks = (masks == obj_ids[:, None, None]).astype(np.uint8)
        else:
            masks = np.zeros((round(height), round(width)), dtype=np.uint8)
            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int32)

        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            "bounds": self._slice_to_tensor(index),
            "transform": ops.convert_to_tensor(np.array(list(transform))),
        }

        if self.task == "semantic_segmentation":
            sample["mask"] = ops.cast(array_to_tensor(masks), self.dtype)

        elif self.task == "object_detection":
            sample["bbox_xyxy"] = ops.convert_to_tensor(boxes_xyxy)
            sample["label"] = ops.convert_to_tensor(labels)

        elif self.task == "instance_segmentation":
            sample["mask"] = array_to_tensor(masks)
            sample["bbox_xyxy"] = ops.convert_to_tensor(boxes_xyxy)
            sample["label"] = ops.convert_to_tensor(labels)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def get_label(self, feature):
        """Get label value to use for rendering a feature."""
        if self.label_name:
            return int(feature[self.label_name])
        return 1


class NonGeoDataset(abc.ABC, PlottingMixin):
    """Abstract base class for datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index):
        """Return an index within the dataset."""

    @abc.abstractmethod
    def __len__(self):
        """Return the length of the dataset."""

    def __str__(self):
        return f"""\
{self.__class__.__name__} Dataset
    type: NonGeoDataset
    size: {len(self)}"""


class NonGeoClassificationDataset(NonGeoDataset):
    """Abstract base class for classification datasets lacking geospatial info.

    This base class is designed for datasets with pre-defined image chips
    separated into separate folders per class (an ``ImageFolder``-style
    layout).
    """

    def __init__(self, root="data", transforms=None, loader=None, is_valid_file=None):
        """Initialize a new NonGeoClassificationDataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            loader: a callable that takes a path to an image and returns a
                PIL Image or numpy array (defaults to a PIL-based loader)
            is_valid_file: a function that takes the path of an image file
                and checks if the file is a valid file
        """
        self.root = str(root)
        self.loader = loader or _pil_loader
        self.classes, self.class_to_idx = _find_classes(self.root)
        self.samples = _make_dataset(self.root, self.class_to_idx, is_valid_file)
        self.imgs = self.samples
        self.tg_transforms = transforms

    def __getitem__(self, index):
        image, label = self._load_image(index)
        sample = {"image": image, "label": label}

        if self.tg_transforms is not None:
            sample = self.tg_transforms(sample)

        return sample

    def __len__(self):
        return len(self.imgs)

    def _load_image(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        array = np.array(img)
        tensor = ops.convert_to_tensor(array.astype("float32"))
        label = ops.convert_to_tensor(np.array(label, dtype="int64"))
        return tensor, label


class IntersectionDataset(GeoDataset):
    """Dataset representing the intersection of two GeoDatasets."""

    def __init__(
        self,
        dataset1,
        dataset2,
        spatial_only=False,
        collate_fn=concat_samples,
        transforms=None,
    ):
        """Initialize a new IntersectionDataset instance.

        When computing the intersection between two datasets that both
        contain model inputs (images) or model outputs (masks), the default
        behavior is to stack the data along the channel dimension. The
        *collate_fn* parameter can be used to change this behavior.
        """
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn
        self.transforms = transforms

        for ds in self.datasets:
            if not isinstance(ds, GeoDataset):
                raise TypeError("IntersectionDataset only supports GeoDatasets")

        dataset2.crs = dataset1.crs
        dataset2.res = dataset1.res

        index1 = dataset1.index.reset_index()
        index2 = dataset2.index.reset_index()
        self.index = gpd.overlay(index1, index2, how="intersection", keep_geom_type=True)

        if self.index.empty:
            raise RuntimeError("Datasets have no spatial intersection")

        columns = ["filepath_1", "filepath_2"]
        self.index.drop(columns=columns, inplace=True, errors="ignore")

        name = "datetime"
        datetime_1 = pd.IntervalIndex(list(self.index.pop("datetime_1")), name=name)
        datetime_2 = pd.IntervalIndex(list(self.index.pop("datetime_2")), name=name)
        self.index.index = datetime_1

        if not spatial_only:
            mint = np.maximum(datetime_1.left, datetime_2.left)
            maxt = np.minimum(datetime_1.right, datetime_2.right)
            valid = maxt >= mint
            mint = mint[valid]
            maxt = maxt[valid]
            self.index = self.index[valid]
            self.index.index = pd.IntervalIndex.from_arrays(
                mint, maxt, closed="both", name="datetime"
            )

            if self.index.empty:
                msg = "Datasets have no temporal intersection. Use `spatial_only=True`"
                msg += " if you want to ignore temporal intersection"
                raise RuntimeError(msg)

    def __getitem__(self, index):
        samples = [ds[index] for ds in self.datasets]

        sample = self.collate_fn(samples)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __str__(self):
        return f"""\
{self.__class__.__name__} Dataset
    type: IntersectionDataset
    bbox: {self.bounds}
    size: {len(self)}"""

    @property
    def crs(self):
        return self.datasets[0].crs

    @crs.setter
    def crs(self, new_crs):
        self.index.to_crs(new_crs, inplace=True)
        self.datasets[0].crs = new_crs
        self.datasets[1].crs = new_crs

    @property
    def res(self):
        return self.datasets[0].res

    @res.setter
    def res(self, new_res):
        self.datasets[0].res = new_res
        self.datasets[1].res = new_res


class UnionDataset(GeoDataset):
    """Dataset representing the union of two GeoDatasets."""

    def __init__(self, dataset1, dataset2, collate_fn=merge_samples, transforms=None):
        """Initialize a new UnionDataset instance.

        When computing the union between two datasets that both contain
        model inputs (images) or model outputs (masks), the default
        behavior is to merge the data to create a single image/mask. The
        *collate_fn* parameter can be used to change this behavior.
        """
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn
        self.transforms = transforms

        for ds in self.datasets:
            if not isinstance(ds, GeoDataset):
                raise TypeError("UnionDataset only supports GeoDatasets")

        dataset2.crs = dataset1.crs
        dataset2.res = dataset1.res

        self.index = pd.concat([dataset1.index, dataset2.index])

    def __getitem__(self, index):
        samples = []
        for ds in self.datasets:
            try:
                samples.append(ds[index])
            except IndexError:
                pass

        if not samples:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        sample = self.collate_fn(samples)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __str__(self):
        return f"""\
{self.__class__.__name__} Dataset
    type: UnionDataset
    bbox: {self.bounds}
    size: {len(self)}"""

    @property
    def crs(self):
        return self.datasets[0].crs

    @crs.setter
    def crs(self, new_crs):
        self.index.to_crs(new_crs, inplace=True)
        self.datasets[0].crs = new_crs
        self.datasets[1].crs = new_crs

    @property
    def res(self):
        return self.datasets[0].res

    @res.setter
    def res(self, new_res):
        self.datasets[0].res = new_res
        self.datasets[1].res = new_res
