"""Rwanda Field Boundary Competition dataset (ported from torchgeo.datasets.rwanda_field_boundary)."""

import glob
import os

import numpy as np
import rasterio
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import which


class RwandaFieldBoundary(NonGeoDataset):
    """Rwanda Field Boundary Competition dataset.

    This dataset contains field boundaries for smallholder farms in eastern
    Rwanda. The Nasa Harvest program funded a team of annotators from
    TaQadam to label Planet imagery for the 2021 growing season for the
    purpose of conducting the Rwanda Field boundary detection Challenge. The
    dataset includes rasterized labeled field boundaries and time series
    satellite imagery from Planet's NICFI program. Planet's basemap imagery
    is provided for six months (March, April, August, October, November and
    December). Note: only fields that were big enough to be differentiated
    on the Planetscope imagery were labeled, only fields that were fully
    contained within the chips were labeled. The paired dataset is provided
    in 256x256 chips for a total of 70 tiles covering 1532 individual
    fields.

    The labels are provided as binary semantic segmentation labels:

    0. No field-boundary
    1. Field-boundary

    If you use this dataset in your research, please cite:

    * https://doi.org/10.34911/RDNT.G580WW

    .. note::

       This dataset requires the following additional library to be
       installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to
         download the dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/nasa_rwanda_field_boundary_competition"

    splits = {"train": 57, "test": 13}
    dates = ("2021_03", "2021_04", "2021_08", "2021_10", "2021_11", "2021_12")
    all_bands = ("B01", "B02", "B03", "B04")
    rgb_bands = ("B03", "B02", "B01")
    classes = ("No field-boundary", "Field-boundary")

    def __init__(
        self, root="data", split="train", bands=all_bands, transforms=None, download=False
    ):
        """Initialize a new RwandaFieldBoundary instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory

        Raises:
            AssertionError: If *split* or *bands* are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert split in self.splits
        assert set(bands) <= set(self.all_bands)

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.download = download

        self._verify()

    def __len__(self):
        return self.splits[self.split]

    def __getitem__(self, index):
        images = []
        for date in self.dates:
            bands_arrs = []
            for band in self.bands:
                path = os.path.join(self.root, "source", self.split, date)
                with rasterio.open(os.path.join(path, f"{index:02}_{band}.tif")) as src:
                    bands_arrs.append(src.read(1).astype("float32"))
            # (H, W, C)
            images.append(np.stack(bands_arrs, axis=-1))
        # (T, H, W, C)
        sample = {"image": ops.convert_to_tensor(np.stack(images))}

        if self.split == "train":
            path = os.path.join(self.root, "labels", self.split)
            with rasterio.open(os.path.join(path, f"{index:02}.tif")) as src:
                mask = src.read(1).astype("int64")
            sample["mask"] = ops.convert_to_tensor(mask)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        path = os.path.join(self.root, "source", self.split, "*", "*.tif")
        expected = len(self.dates) * self.splits[self.split] * len(self.all_bands)
        if len(glob.glob(path)) == expected:
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)
        azcopy = which("azcopy")
        azcopy("sync", self.url, self.root, "--recursive=true")

    def plot(self, sample, show_titles=True, suptitle=None, time_step=0):
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

        ncols = 1
        for key in ("mask", "prediction"):
            if key in sample:
                ncols += 1

        fig, axs = plt.subplots(ncols=ncols, squeeze=False)

        image = ops.convert_to_numpy(sample["image"])[time_step]
        image = np.take(image, rgb_indices, axis=-1)
        image = np.clip(image / 2000, 0, 1)
        axs[0, 0].imshow(image)
        axs[0, 0].axis("off")
        if show_titles:
            axs[0, 0].set_title(f"t={time_step}")

        if "mask" in sample:
            axs[0, 1].imshow(ops.convert_to_numpy(sample["mask"]))
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Mask")

        if "prediction" in sample:
            axs[0, 2].imshow(ops.convert_to_numpy(sample["prediction"]))
            axs[0, 2].axis("off")
            if show_titles:
                axs[0, 2].set_title("Prediction")

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
