"""Copernicus-Bench abstract base class (ported from torchgeo.datasets.copernicus.base)."""

import os
import re
from abc import ABC, abstractmethod

import numpy as np
import rasterio as rio
from einops import rearrange
from keras import ops
from pyproj import Transformer

from ..errors import DatasetNotFoundError, RGBBandsMissingError
from ..geo import NonGeoDataset
from ..utils import (
    array_to_tensor,
    disambiguate_timestamp,
    download_and_extract_archive,
    extract_archive,
    quantile_normalization,
)


class CopernicusBenchBase(NonGeoDataset, ABC):
    """Abstract base class for all Copernicus-Bench datasets.

    If you use this dataset in your research, please cite:
    https://arxiv.org/abs/2503.11849
    """

    @property
    @abstractmethod
    def url(self):
        """Download URL."""

    #: SHA256 checksum.
    sha256 = None

    #: Zip file name.
    zipfile = None

    #: Subdirectory containing split files.
    directory = None

    #: Filename format of split files.
    filename = "{}.csv"

    #: Mask dtype to cast to, either "int64" for classification or
    #: "float32" for regression.
    dtype = "int64"

    #: Regular expression used to extract date from filename.
    filename_regex = ".*"

    #: Date format string used to parse date from filename.
    date_format = "%Y%m%dT%H%M%S"

    #: Matplotlib color map for semantic segmentation and change detection plots.
    cmap = None

    #: List of classes for classification, semantic segmentation, and change detection.
    classes = ()

    def __init__(
        self,
        root="data",
        split="train",
        bands=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new CopernicusBenchBase instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its
                target as entry and returns a transformed version.
            download: If True, download dataset and store it in the root
                directory.
            checksum: If True, verify the checksum of the downloaded files
                (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        import pandas as pd

        self.root = root
        self.split = split
        self.bands = bands or self.all_bands
        self.band_indices = [self.all_bands.index(i) + 1 for i in self.bands]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath, header=None)[0]

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        """Load an image and metadata.

        Returns:
            An image sample, channels-last (``H x W x C``).
        """
        sample = {}
        with rio.open(path) as f:
            image = f.read(self.band_indices).astype(np.float32)
            # (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))
            sample["image"] = ops.convert_to_tensor(image)

            if f.transform != rio.Affine.identity():
                x = (f.bounds.left + f.bounds.right) / 2
                y = (f.bounds.bottom + f.bounds.top) / 2
                transformer = Transformer.from_crs(f.crs, "epsg:4326", always_xy=True)
                lon, lat = transformer.transform(x, y)
                sample["lat"] = ops.convert_to_tensor(lat)
                sample["lon"] = ops.convert_to_tensor(lon)

            if match := re.match(self.filename_regex, os.path.basename(path)):
                if "date" in match.groupdict():
                    date_str = match.group("date")
                    mint, maxt = disambiguate_timestamp(date_str, self.date_format)
                    time = (mint.timestamp() + maxt.timestamp()) / 2
                    sample["time"] = ops.convert_to_tensor(time)
                elif "start" in match.groupdict() and "stop" in match.groupdict():
                    start = match.group("start")
                    stop = match.group("stop")
                    mint, _ = disambiguate_timestamp(start, self.date_format)
                    _, maxt = disambiguate_timestamp(stop, self.date_format)
                    time = (mint.timestamp() + maxt.timestamp()) / 2
                    sample["time"] = ops.convert_to_tensor(time)

        return sample

    def _load_mask(self, path):
        """Load a target mask."""
        sample = {}
        with rio.open(path) as f:
            sample["mask"] = ops.cast(array_to_tensor(f.read(1)), self.dtype)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        path = os.path.join(self.root, self.directory, self.filename.format(self.split))
        if os.path.exists(path):
            return

        if os.path.exists(os.path.join(self.root, self.zipfile)):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _extract(self):
        extract_archive(os.path.join(self.root, self.zipfile))

    def _download(self):
        sha256 = self.sha256 if self.checksum else None
        download_and_extract_archive(self.url, self.root, sha256=sha256)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        import matplotlib.pyplot as plt

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        # Static -> time series. Channels-last: (H, W, C) or (T, H, W, C).
        images = sample["image"]
        if len(images.shape) == 3:
            images = ops.expand_dims(images, axis=0)

        ncols = images.shape[0]
        if "mask" in sample:
            ncols += 1
            if "prediction" in sample:
                ncols += 1

        fig, ax = plt.subplots(ncols=ncols, squeeze=False)

        title = "Image"
        if "label" in sample:
            label_arr = ops.convert_to_numpy(sample["label"])
            if label_arr.ndim == 0:
                # Multiclass classification
                label = self.classes[int(label_arr)]
                if "prediction" in sample:
                    prediction = self.classes[
                        int(ops.convert_to_numpy(sample["prediction"]))
                    ]
            else:
                # Multilabel classification
                label = label_arr.nonzero()[0]
                if "prediction" in sample:
                    prediction = ops.convert_to_numpy(
                        sample["prediction"]
                    ).nonzero()[0]

            title = f"Label: {label}"
            if "prediction" in sample:
                title += f"\nPrediction: {prediction}"

        images = ops.take(images, ops.convert_to_tensor(rgb_indices), axis=-1)
        if set(self.rgb_bands) <= {"VV", "VH", "HH", "HV"}:
            # SAR
            vv = images[..., 0]
            vh = images[..., 1]
            images = ops.stack([vv, vh, (vv + vh) / 2], axis=-1)
            images = quantile_normalization(images)

        images = quantile_normalization(images)
        images = ops.convert_to_numpy(images)
        for i in range(len(images)):
            ax[0, i].imshow(images[i])
            ax[0, i].axis("off")
            if show_titles:
                ax[0, i].set_title(title)

        if "mask" in sample:
            kwargs = {"cmap": self.cmap}
            if self.classes:
                kwargs |= {
                    "vmin": 0,
                    "vmax": len(self.classes) - 1,
                    "interpolation": "none",
                }
            mask = ops.convert_to_numpy(sample["mask"])
            ax[0, i + 1].imshow(mask, **kwargs)
            ax[0, i + 1].axis("off")
            if show_titles:
                ax[0, i + 1].set_title("Mask")

            if "prediction" in sample:
                prediction = ops.convert_to_numpy(sample["prediction"])
                ax[0, i + 2].imshow(prediction, **kwargs)
                ax[0, i + 2].axis("off")
                if show_titles:
                    ax[0, i + 2].set_title("Prediction")

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()

        return fig
