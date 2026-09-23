"""Cloud Cover Detection Challenge dataset (ported from torchgeo.datasets.cloud_cover)."""

import os

import numpy as np
import pandas as pd
import rasterio
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import which


class CloudCoverDetection(NonGeoDataset):
    """Sentinel-2 Cloud Cover Segmentation Dataset.

    This training dataset was generated as part of a `crowdsourcing competition
    <https://www.drivendata.org/competitions/83/cloud-cover/>`_ on DrivenData.org, and
    later on was validated using a team of expert annotators. See `this website
    <https://source.coop/radiantearth/cloud-cover-detection-challenge>`__
    for dataset details.

    The dataset consists of Sentinel-2 satellite imagery and corresponding cloudy
    labels stored as GeoTiffs. There are 22,728 chips in the training data,
    collected between 2018 and 2020.

    Each chip has:

    * 4 multi-spectral bands from Sentinel-2 L2A product. The four bands are
      [B02, B03, B04, B08] (refer to Sentinel-2 documentation for more
      information about the bands).
    * Label raster for the corresponding source tile representing a binary
      classification for if the pixel is a cloud or not.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/RDNT.HFQ6M7

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/ref_cloud_cover_detection_challenge_v1/final"
    all_bands = ("B02", "B03", "B04", "B08")
    rgb_bands = ("B04", "B03", "B02")
    splits = {"train": "public", "test": "private"}

    def __init__(self, root="data", split="train", bands=all_bands, transforms=None, download=False):
        """Initiatlize a CloudCoverDetection instance.

        Args:
            root: root directory where dataset can be found
            split: 'train' or 'test'
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
        self.directory = os.path.join(self.root, self.splits[self.split])
        self.bands = bands
        self.transforms = transforms
        self.download = download

        self.csv = os.path.join(self.directory, f"{self.split}_metadata.csv")
        self._verify()

        self.metadata = pd.read_csv(self.csv)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        chip_id = str(self.metadata.iat[index, 0])
        image = self._load_image(chip_id)
        label = self._load_target(chip_id)
        sample = {"image": image, "mask": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, chip_id):
        path = os.path.join(self.directory, f"{self.split}_features", chip_id)
        images = []
        for band in self.bands:
            with rasterio.open(os.path.join(path, f"{band}.tif")) as src:
                images.append(src.read(1).astype("float32"))
        return ops.convert_to_tensor(np.stack(images, axis=-1))

    def _load_target(self, chip_id):
        path = os.path.join(self.directory, f"{self.split}_labels")
        with rasterio.open(os.path.join(path, f"{chip_id}.tif")) as src:
            return ops.convert_to_tensor(src.read(1).astype("int64"))

    def _verify(self):
        """Verify the integrity of the dataset."""
        if os.path.exists(self.csv):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the dataset."""
        directory = self.directory
        os.makedirs(directory, exist_ok=True)
        url = f"{self.url}/{self.splits[self.split]}"
        azcopy = which("azcopy")
        azcopy("sync", url, directory, "--recursive=true")

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

        if "prediction" in sample:
            prediction = ops.convert_to_numpy(sample["prediction"])
            ncols = 3
        else:
            ncols = 2

        image = ops.convert_to_numpy(ops.take(sample["image"], rgb_indices, axis=-1)) / 3000
        mask = ops.convert_to_numpy(sample["mask"])

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if "prediction" in sample:
            axs[2].imshow(prediction)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
