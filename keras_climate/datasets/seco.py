"""Sentinel 2 imagery from the Seasonal Contrast paper (ported from
torchgeo.datasets.seco).
"""

import os
import random

import numpy as np
import rasterio
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive, quantile_normalization


class SeasonalContrastS2(NonGeoDataset):
    """Sentinel 2 imagery from the Seasonal Contrast paper.

    The `Seasonal Contrast imagery <https://github.com/ServiceNow/seasonal-contrast>`_
    dataset contains Sentinel 2 imagery patches sampled from different points in time
    around the 10k most populated cities on Earth.

    Dataset features:

    * Two versions: 100K and 1M patches
    * 12 band Sentinel 2 imagery from 5 points in time at each location

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/pdf/2103.16607
    """

    all_bands = (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    )
    rgb_bands = ("B4", "B3", "B2")

    metadata = {
        "100k": {
            "url": "https://zenodo.org/records/4728033/files/seco_100k.zip?download=1",
            "md5": "ebf2d5e03adc6e657f9a69a20ad863e0",
            "filename": "seco_100k.zip",
            "directory": "seasonal_contrast_100k",
        },
        "1m": {
            "url": "https://zenodo.org/records/4728033/files/seco_1m.zip?download=1",
            "md5": "187963d852d4d3ce6637743ec3a4bd9e",
            "filename": "seco_1m.zip",
            "directory": "seasonal_contrast_1m",
        },
    }

    def __init__(
        self,
        root="data",
        version="100k",
        seasons=1,
        bands=rgb_bands,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new SeasonalContrastS2 instance.

        Args:
            root: root directory where dataset can be found
            version: one of "100k" or "1m" for the version of the dataset to use
            seasons: number of seasonal patches to sample per location, 1--5
            bands: list of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``version`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert version in self.metadata
        assert seasons in range(5)
        for band in bands:
            assert band in self.all_bands

        self.root = root
        self.version = version
        self.seasons = seasons
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

    def __getitem__(self, index):
        """Return an index within the dataset.

        Returns:
            sample with an "image" in H x W x SC format where S is the number of
            seasons
        """
        root = os.path.join(
            self.root, self.metadata[self.version]["directory"], f"{index:06}"
        )
        subdirs = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
        subdirs = random.sample(subdirs, self.seasons)

        images = [self._load_patch(root, subdir) for subdir in subdirs]

        sample = {"image": ops.concatenate(images, axis=-1)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return (10**5 if self.version == "100k" else 10**6) // 5

    def _load_patch(self, root, subdir):
        """Load a single image patch.

        Returns:
            the image with the subset of bands specified by ``self.bands``, shape
            (H, W, C)
        """
        from PIL import Image

        all_data = []
        for band in self.bands:
            fn = os.path.join(root, subdir, f"{band}.tif")
            with rasterio.open(fn) as f:
                band_data = f.read(1).astype(np.float32)
                height, width = band_data.shape
                size = min(height, width)
                if size < 264:
                    pil_image = Image.fromarray(band_data)
                    resample = Image.Resampling.BILINEAR
                    band_data = np.array(
                        pil_image.resize((264, 264), resample=resample)
                    )
                all_data.append(band_data)
        image = np.stack(all_data, axis=-1)
        return ops.convert_to_tensor(image)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        directory_path = os.path.join(
            self.root, self.metadata[self.version]["directory"]
        )
        if os.path.exists(directory_path):
            return

        # Check if the zip files have already been downloaded
        zip_path = os.path.join(self.root, self.metadata[self.version]["filename"])
        if os.path.exists(zip_path):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self):
        """Download the dataset."""
        download_url(
            self.metadata[self.version]["url"],
            self.root,
            filename=self.metadata[self.version]["filename"],
            md5=self.metadata[self.version]["md5"] if self.checksum else None,
        )

    def _extract(self):
        """Extract the dataset."""
        extract_archive(
            os.path.join(self.root, self.metadata[self.version]["filename"])
        )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
            ValueError: if sample contains a "prediction" key
        """
        import matplotlib.pyplot as plt

        if "prediction" in sample:
            raise ValueError("This dataset doesn't support plotting predictions")

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        fig, axes = plt.subplots(ncols=self.seasons, figsize=(20, 4))
        if self.seasons == 1:
            axes = [axes]

        for i in range(self.seasons):
            indices = [idx + i * len(self.bands) for idx in rgb_indices]
            image = ops.take(sample["image"], indices, axis=-1)
            image = quantile_normalization(image)

            axes[i].imshow(ops.convert_to_numpy(image))
            axes[i].axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
