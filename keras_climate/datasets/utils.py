"""Common dataset utilities (ported from torchgeo.datasets.utils).

Ported to be framework-agnostic: sample values are Keras tensors (produced
via ``keras.ops.convert_to_tensor``) instead of ``torch.Tensor``, so this
module works unchanged under the TensorFlow, JAX, and PyTorch Keras 3
backends.
"""

from __future__ import annotations

import bz2
import contextlib
import fnmatch
import glob
import hashlib
import importlib
import os
import pathlib
import shutil
import subprocess
import tarfile
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely.affinity
from keras import ops
from pandas import Timedelta, Timestamp
from rasterio.features import shapes, sieve
from shapely import MultiPolygon, Polygon, box

from .errors import DependencyNotFoundError

#: Slice to index a GeoDataset.
#:
#: Can handle several different forms, such as:
#:
#:     ds[xmin:xmax:xres, ymin:ymax:yres]
#:     ds[:, :, tmin:tmax:tres]
#:     ds[xmin:xmax, ymin:ymax, tmin:tmax]
#:
#: All values are optional and default to the spatiotemporal extent of the dataset.
GeoSlice = "slice | tuple[slice] | tuple[slice, slice] | tuple[slice, slice, slice]"

#: Path-like object. Most datasets can handle any kind of path-like object,
#: and some can support a list of paths.
Path = "str | os.PathLike[str]"

#: Sample dictionary returned by a dataset. Keys typically include:
#:
#: * image: input image
#: * mask: expected output semantic segmentation mask
#: * label: expected output classification or regression label
#: * bbox_xyxy: expected output bounding box in (x1, y1, x2, y2) format
#: * prediction: predicted output
#:
#: Values are Keras tensors.
Sample = dict


@dataclass(frozen=True)
class BoundingBox:
    """Data class for indexing spatiotemporal data.

    .. deprecated:: Use GeoSlice or shapely.Polygon instead.
    """

    #: western boundary
    minx: float
    #: eastern boundary
    maxx: float
    #: southern boundary
    miny: float
    #: northern boundary
    maxy: float
    #: earliest boundary
    mint: datetime
    #: latest boundary
    maxt: datetime

    def __post_init__(self):
        if self.minx > self.maxx:
            raise ValueError(
                f"Bounding box is invalid: 'minx={self.minx}' > 'maxx={self.maxx}'"
            )
        if self.miny > self.maxy:
            raise ValueError(
                f"Bounding box is invalid: 'miny={self.miny}' > 'maxy={self.maxy}'"
            )
        if self.mint > self.maxt:
            raise ValueError(
                f"Bounding box is invalid: 'mint={self.mint}' > 'maxt={self.maxt}'"
            )

    def __getitem__(self, key):
        return [self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt][key]

    def __iter__(self):
        yield from [self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt]

    def __contains__(self, other):
        return (
            (self.minx <= other.minx <= self.maxx)
            and (self.minx <= other.maxx <= self.maxx)
            and (self.miny <= other.miny <= self.maxy)
            and (self.miny <= other.maxy <= self.maxy)
            and (self.mint <= other.mint <= self.maxt)
            and (self.mint <= other.maxt <= self.maxt)
        )

    def __or__(self, other):
        return BoundingBox(
            min(self.minx, other.minx),
            max(self.maxx, other.maxx),
            min(self.miny, other.miny),
            max(self.maxy, other.maxy),
            min(self.mint, other.mint),
            max(self.maxt, other.maxt),
        )

    def __and__(self, other):
        try:
            return BoundingBox(
                max(self.minx, other.minx),
                min(self.maxx, other.maxx),
                max(self.miny, other.miny),
                min(self.maxy, other.maxy),
                max(self.mint, other.mint),
                min(self.maxt, other.maxt),
            )
        except ValueError:
            raise ValueError(f"Bounding boxes {self} and {other} do not overlap")

    @property
    def area(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    @property
    def volume(self):
        return self.area * (self.maxt - self.mint)

    def intersects(self, other):
        return (
            self.minx <= other.maxx
            and self.maxx >= other.minx
            and self.miny <= other.maxy
            and self.maxy >= other.miny
            and self.mint <= other.maxt
            and self.maxt >= other.mint
        )

    def split(self, proportion, horizontal=True):
        if not (0.0 < proportion < 1.0):
            raise ValueError("Input proportion must be between 0 and 1.")

        if horizontal:
            w = self.maxx - self.minx
            splitx = self.minx + w * proportion
            bbox1 = BoundingBox(
                self.minx, splitx, self.miny, self.maxy, self.mint, self.maxt
            )
            bbox2 = BoundingBox(
                splitx, self.maxx, self.miny, self.maxy, self.mint, self.maxt
            )
        else:
            h = self.maxy - self.miny
            splity = self.miny + h * proportion
            bbox1 = BoundingBox(
                self.minx, self.maxx, self.miny, splity, self.mint, self.maxt
            )
            bbox2 = BoundingBox(
                self.minx, self.maxx, splity, self.maxy, self.mint, self.maxt
            )

        return bbox1, bbox2


class Executable:
    """Command-line executable."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return subprocess.run((self.name, *args), check=True, **kwargs)


def check_integrity(fpath, md5=None, **kwargs):
    """Check the integrity of a file."""
    if not os.path.isfile(fpath):
        return False

    kwargs["md5"] = md5

    for algorithm, checksum in kwargs.items():
        if checksum:
            with open(fpath, "rb") as f:
                return hashlib.file_digest(f, algorithm).hexdigest() == checksum

    return True


def extract_archive(from_path, to_path=None, remove_finished=False):
    """Extract an archive."""
    to_path = to_path or os.path.dirname(from_path)
    suffixes = pathlib.Path(from_path).suffixes

    if suffixes[-1] == ".zip":
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    elif suffixes[-1] == ".bz2" and ".tar" not in suffixes:
        stem = pathlib.Path(from_path).stem
        to_path = os.path.join(to_path, stem)
        with bz2.open(from_path, "rb") as src, open(to_path, "wb") as dst:
            dst.write(src.read())
    else:
        with tarfile.open(from_path, "r") as t:
            t.extractall(to_path, filter="data")

    if remove_finished:
        os.remove(from_path)

    return to_path


def download_url(url, root, filename=None, md5=None, max_redirect_hops=3, **kwargs):
    """Download a file from a url and place it in root."""
    if not filename:
        filename = os.path.basename(url)

    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)

    fpath = os.path.join(root, filename)
    if not check_integrity(fpath, md5, **kwargs):
        request = urllib.request.Request(url, headers={"User-Agent": "keras_climate"})
        tmp = f"{fpath}.tmp"
        try:
            with urllib.request.urlopen(request) as response:
                total = response.headers.get("Content-Length")
                expected = int(total) if total else None
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(response, f)
            if expected is not None:
                actual = os.path.getsize(tmp)
                if actual != expected:
                    raise RuntimeError(
                        f"Downloaded file '{fpath}' is incomplete: expected "
                        f"{expected} bytes but got {actual}."
                    )
            os.replace(tmp, fpath)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if not check_integrity(fpath, md5, **kwargs):
            raise RuntimeError(f"Downloaded file '{fpath}' is corrupted.")


def download_and_extract_archive(
    url,
    download_root,
    extract_root=None,
    filename=None,
    md5=None,
    remove_finished=False,
    **kwargs,
):
    """Download and extract a remote archive."""
    download_root = os.path.expanduser(download_root)
    extract_root = extract_root or download_root
    filename = filename or os.path.basename(url)
    from_path = os.path.join(download_root, filename)

    download_url(url, download_root, filename, md5, 3, **kwargs)
    extract_archive(from_path, extract_root, remove_finished)


def disambiguate_timestamp(date_str, format):
    """Disambiguate partial timestamps.

    Datasets store the timestamp of each file in a pandas IntervalIndex. If
    the full timestamp isn't known, a file could represent a range of time.
    This returns the maximum possible range of timestamps that ``date_str``
    could belong to, by parsing ``format`` to determine its precision.
    """
    mint = pd.to_datetime(date_str, format=format)
    format = format.replace("%%", "")

    if not any(f"%{c}" in format for c in "yYcxG"):
        return Timestamp.min, Timestamp.max
    elif not any(f"%{c}" in format for c in "bBmjUWcxV"):
        maxt = Timestamp(year=mint.year + 1, month=1, day=1)
    elif not any(f"%{c}" in format for c in "aAwdjcxV"):
        if mint.month == 12:
            maxt = Timestamp(year=mint.year + 1, month=1, day=1)
        else:
            maxt = Timestamp(year=mint.year, month=mint.month + 1, day=1)
    elif not any(f"%{c}" in format for c in "HIcX"):
        maxt = mint + Timedelta(days=1)
    elif not any(f"%{c}" in format for c in "McX"):
        maxt = mint + Timedelta(hours=1)
    elif not any(f"%{c}" in format for c in "ScX"):
        maxt = mint + Timedelta(minutes=1)
    elif not any(f"%{c}" in format for c in "f"):
        maxt = mint + Timedelta(seconds=1)
    else:
        maxt = mint + Timedelta(microseconds=1)

    maxt -= Timedelta(microseconds=1)

    return mint, maxt


@contextlib.contextmanager
def working_dir(dirname, create=False):
    """Context manager for changing directories."""
    if create:
        os.makedirs(dirname, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(dirname)

    try:
        yield
    finally:
        os.chdir(cwd)


def _list_dict_to_dict_list(samples):
    collated = {}
    for sample in samples:
        for key, value in sample.items():
            if key not in collated:
                collated[key] = []
            collated[key].append(value)
    return collated


def _dict_list_to_list_dict(sample):
    uncollated = [{} for _ in range(max(map(len, sample.values())))]
    for key, values in sample.items():
        for i, value in enumerate(values):
            uncollated[i][key] = value
    return uncollated


def pad_across_batches(batch, padding_length, padding_value=0.0):
    """Custom time-series collate fn to handle variable length sequences."""
    collated = {}
    images = [ops.convert_to_numpy(sample["image"]) for sample in batch]
    feature_shape = images[0].shape[1:]

    padded_images = np.full(
        (len(batch), padding_length, *feature_shape),
        padding_value,
        dtype=images[0].dtype,
    )

    truncated = 0
    lengths = []
    for i, img in enumerate(images):
        seq_len = img.shape[0]
        if seq_len > padding_length:
            padded_images[i, :padding_length] = img[:padding_length]
            truncated += 1
            lengths.append(padding_length)
        else:
            padded_images[i, :seq_len] = img
            lengths.append(seq_len)

    if truncated > 0:
        warnings.warn(f"Truncated {truncated} sequences to length {padding_length}.")

    collated["image"] = ops.convert_to_tensor(padded_images)
    collated["length"] = ops.convert_to_tensor(np.array(lengths, dtype="int64"))
    if "mask" in batch[0]:
        collated["mask"] = ops.stack([sample["mask"] for sample in batch])
    if "bbox_xyxy" in batch[0]:
        collated["bbox_xyxy"] = ops.stack([sample["bbox_xyxy"] for sample in batch])
    if "label" in batch[0]:
        collated["label"] = ops.stack([sample["label"] for sample in batch])

    return collated


def stack_samples(samples):
    """Stack a list of samples along a new axis."""
    uncollated = _list_dict_to_dict_list(samples)
    collated = {}
    for key, value in uncollated.items():
        collated[key] = ops.stack(value)
    return collated


def concat_samples(samples):
    """Concatenate a list of samples along an existing axis.

    Concatenates along the last axis (the channel axis for channels-last
    ``image``/``mask`` samples). For 1-D metadata values (e.g. ``bounds``,
    ``transform``) the last axis is the only axis, so this matches the
    previous (and torchgeo's channels-first) axis-0 behavior for those keys.
    """
    uncollated = _list_dict_to_dict_list(samples)
    collated = {}
    for key, value in uncollated.items():
        collated[key] = ops.concatenate(value, axis=-1)
    return collated


def merge_samples(samples):
    """Merge a list of samples, taking the elementwise maximum for shared keys."""
    collated = {}
    for sample in samples:
        for key, value in sample.items():
            if key in collated:
                # Take the maximum so that nodata values (zeros) get replaced
                # by data values whenever possible
                collated[key] = ops.maximum(collated[key], value)
            else:
                collated[key] = value
    return collated


def unbind_samples(sample):
    """Reverse of :func:`stack_samples`."""
    uncollated = {}
    for key, values in sample.items():
        uncollated[key] = [values[i] for i in range(values.shape[0])]
    return _dict_list_to_list_dict(uncollated)


def rasterio_loader(path):
    """Load an image file using rasterio."""
    with rasterio.open(path) as f:
        array = f.read().astype(np.int32)
        # NonGeoClassificationDataset expects images returned channels-last (HWC)
        array = array.transpose(1, 2, 0)
    return array


def sort_sentinel2_bands(x):
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split("_")[-1]
    x = os.path.splitext(x)[0]
    if x == "B8A":
        x = "B08A"
    return x


def draw_semantic_segmentation_masks(image, mask, alpha=0.5, colors=None):
    """Overlay a semantic segmentation mask onto an image.

    Args:
        image: tensor of shape (h, w, 3) and dtype uint8.
        mask: tensor of shape (h, w) with pixel values representing classes.
        alpha: alpha blend factor.
        colors: list of RGB int tuples used to render each class.

    Returns:
        A uint8 numpy array of the overlaid image.
    """
    image = ops.convert_to_numpy(image).astype(np.uint8)
    mask = ops.convert_to_numpy(mask)
    colors = colors or []
    out = image.copy().astype(np.float64)
    for class_idx, color in enumerate(colors):
        class_mask = mask == class_idx
        if not class_mask.any():
            continue
        color_arr = np.array(color, dtype=np.float64)
        out[class_mask] = (1 - alpha) * out[class_mask] + alpha * color_arr
    return out.astype(np.uint8)


def rgb_to_mask(rgb, colors):
    """Converts an RGB colormap mask to a integer mask."""
    assert len(colors) <= 256

    h, w = rgb.shape[:2]
    mask = np.zeros(shape=(h, w), dtype=np.uint8)
    for i, c in enumerate(colors):
        cmask = rgb == c
        if isinstance(cmask, np.ndarray):
            mask[cmask.all(axis=-1)] = i
    return mask


def percentile_normalization(img, lower=2, upper=98, axis=None, nodata=0):
    """Applies percentile normalization to an input image.

    .. deprecated:: Use :func:`quantile_normalization` instead.
    """
    if (img == nodata).all():
        return img

    assert lower < upper
    lower_percentile = np.percentile(img[img != nodata], lower, axis=axis)
    upper_percentile = np.percentile(img[img != nodata], upper, axis=axis)
    img_normalized = np.clip(
        (img - lower_percentile) / (upper_percentile - lower_percentile + 1e-5), 0, 1
    )
    return img_normalized


def quantile_normalization(img, lower=0.02, upper=0.98, nodata=0, dim=None):
    """Normalize and clip an input image to a specific quantile range."""
    array = ops.convert_to_numpy(img)
    if (array == nodata).all():
        return img

    valid = array[array != nodata]
    lo = np.quantile(valid, lower, axis=dim, method="higher")
    hi = np.quantile(valid, upper, axis=dim, method="lower")
    normalized = (array - lo) / (hi - lo + 1e-5)
    normalized = np.clip(normalized, 0, 1)
    return ops.convert_to_tensor(normalized)


def array_to_tensor(array):
    """Converts a :class:`numpy.ndarray` to a Keras tensor.

    Casts uint16/uint32 numpy arrays (unsupported by most Keras backends) to
    an appropriate signed dtype without loss of precision before conversion.
    """
    if array.dtype == np.uint16:
        array = array.astype(np.int32)
    elif array.dtype == np.uint32:
        array = array.astype(np.int64)
    return ops.convert_to_tensor(array)


def lazy_import(name):
    """Lazy import of *name*."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pkg = name.split(".")[0].replace("_", "-")
        msg = f"""\
{pkg} is not installed and is required to use this feature. Either run:

$ pip install {pkg}

to install just this dependency, or:

$ pip install keras_climate[datasets]

to install all optional dependencies."""
        raise DependencyNotFoundError(msg) from None


def which(name):
    """Search for executable *name*."""
    if cmd := shutil.which(name):
        return Executable(cmd)
    else:
        msg = f"{name} is not installed and is required to use this dataset."
        raise DependencyNotFoundError(msg) from None


def convert_poly_coords(geom, affine_obj, inverse=False):
    """Convert geocoordinates to pixel coordinates and vice versa."""
    if inverse:
        affine_obj = ~affine_obj

    return shapely.affinity.affine_transform(
        geom,
        [
            affine_obj.a,
            affine_obj.b,
            affine_obj.d,
            affine_obj.e,
            affine_obj.xoff,
            affine_obj.yoff,
        ],
    )


def _list_vsi_files(root):
    try:
        entries = pyogrio.vsi_listtree(str(root))
    except NotADirectoryError:
        return [str(root)]
    except FileNotFoundError:
        return []
    return [e for e in entries if not e.endswith("/")]


def find_files(path, filename_glob="*"):
    """Return all files under *path* that match *filename_glob*.

    Supports local directories, individual files, and VSI paths such as
    cloud storage buckets and local archives (zip, tar, etc.).
    """
    files = set()
    if os.path.isdir(path):
        pathname = os.path.join(path, "**", filename_glob)
        files = set(glob.iglob(pathname, recursive=True))
    elif os.path.isfile(path) and fnmatch.fnmatch(str(path), f"*{filename_glob}"):
        files = {str(path)}
    elif str(path).startswith("/vsi"):
        all_files = _list_vsi_files(path)
        files = {
            f for f in all_files if fnmatch.fnmatch(os.path.basename(f), filename_glob)
        }
    return sorted(files)


def _clean_binary_mask(mask, threshold=1):
    """Convert any rasterio mask to a clean binary mask (uint8 0 or 255)."""
    if mask.ndim == 3:
        combined = np.any(mask >= threshold, axis=0)
    else:
        combined = mask >= threshold

    return (combined.astype(np.uint8)) * 255


def _binary_mask_to_polygon(mask, transform, raster_resolution_x):
    """Vectorize a binary raster mask into a polygon."""
    max_hole_size = min(int(mask.size * 0.002), 800) or 1
    sieved_mask = sieve(mask, max_hole_size)

    geoms = [g for g, v in shapes(sieved_mask, transform=transform) if v > 0]

    vector_mask = MultiPolygon(
        [
            Polygon(feature["coordinates"][0], feature["coordinates"][1:])
            for feature in geoms
        ]
    )
    simplification_tolerance = 2 * raster_resolution_x
    return vector_mask.simplify(simplification_tolerance)


def get_valid_footprint_from_datasource(src):
    """Compute the valid (non-NoData) data footprint of a raster in its CRS."""
    valid_data_mask = src.dataset_mask()
    binary_mask = _clean_binary_mask(valid_data_mask)

    if (binary_mask == 255).all():
        return box(*src.bounds)

    if (binary_mask == 0).all():
        warnings.warn(
            "All pixels are nodata; returning an empty geometry.",
            UserWarning,
            stacklevel=2,
        )
        return Polygon()

    res_x = src.res[0] if isinstance(src.res, tuple) else src.res
    return _binary_mask_to_polygon(
        mask=binary_mask, transform=src.transform, raster_resolution_x=res_x
    )
