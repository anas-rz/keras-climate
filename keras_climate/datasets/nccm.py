"""Northeastern China Crop Map Dataset (ported from torchgeo.datasets.nccm)."""

import os

import numpy as np
from keras import ops
from matplotlib.colors import ListedColormap

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import download_url


class NCCM(RasterDataset):
    """The Northeastern China Crop Map Dataset.

    Link: https://www.nature.com/articles/s41597-021-00827-9

    This dataset produced annual 10-m crop maps of the major crops (maize,
    soybean, and rice) in Northeast China from 2017 to 2019, using
    hierarchial mapping strategies, random forest classifiers, interpolated
    and smoothed 10-day Sentinel-2 time series data and optimized features
    from spectral, temporal and textural characteristics of the land
    surface. The resultant maps have high overall accuracies (OA) based on
    ground truth data. The dataset contains information specific to three
    years: 2017, 2018, 2019.

    The dataset contains 5 classes:

    0. paddy rice
    1. maize
    2. soybean
    3. others crops and lands
    4. nodata

    Dataset format:

    * Three .TIF files containing the labels
    * JavaScript code to download images from the dataset.

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.1038/s41597-021-00827-9
    """

    filename_regex = r"CDL(?P<date>\d{4})_clip"
    filename_glob = "CDL*.*"

    date_format = "%Y"
    is_image = False
    urls = {
        2019: "https://api.figshare.com/v2/file/download/25070540",
        2018: "https://api.figshare.com/v2/file/download/25070624",
        2017: "https://api.figshare.com/v2/file/download/25070582",
    }
    md5s = {
        2019: "0d062bbd42e483fdc8239d22dba7020f",
        2018: "b3bb4894478d10786aa798fb11693ec1",
        2017: "d047fbe4a85341fa6248fd7e0badab6c",
    }
    fnames = {
        2019: "CDL2019_clip.tif",
        2018: "CDL2018_clip1.tif",
        2017: "CDL2017_clip.tif",
    }

    cmap = ListedColormap(
        [(0, 1, 0, 1), (1, 0, 0, 1), (1, 1, 0, 1), (0.5, 0.5, 0.5, 1), (1, 1, 1, 1)]
    )

    def __init__(
        self,
        paths="data",
        crs=None,
        res=None,
        years=None,
        transforms=None,
        cache=True,
        download=False,
        checksum=True,
        time_series=False,
    ):
        """Initialize a new dataset.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres)
                format (defaults to the resolution of the first file found)
            years: list of years for which to use nccm layers
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)
            time_series: if True, stack data along the time series dimension
                [T, H, W, C]. If False, merge data into a [H, W, C] mosaic.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        if years is None:
            years = [2019]
        assert set(years) <= self.md5s.keys(), (
            f"NCCM data product only exists for the following years: {list(self.md5s.keys())}."
        )
        self.paths = paths
        self.years = years
        self.download = download
        self.checksum = checksum

        self._verify()
        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

    def __getitem__(self, index):
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice."""
        sample = super().__getitem__(index)

        # Convert nodata class (15) to 4 so there are no gaps in our ordinal mapping
        mask = ops.convert_to_numpy(sample["mask"])
        mask = np.where(mask == 15, 4, mask)
        sample["mask"] = ops.convert_to_tensor(mask)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        if self.files:
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the dataset."""
        paths = self.paths
        for year in self.years:
            download_url(
                self.urls[year],
                paths,
                filename=self.fnames[year],
                md5=self.md5s[year] if self.checksum else None,
            )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        mask = ops.convert_to_numpy(sample["mask"])
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = ops.convert_to_numpy(sample["prediction"])
            ncols = 2

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 4, 4), squeeze=False)
        kwargs = {"cmap": self.cmap, "vmin": 0, "vmax": 4, "interpolation": "none"}

        axs[0, 0].imshow(mask, **kwargs)
        axs[0, 0].axis("off")

        if show_titles:
            axs[0, 0].set_title("Mask")

        if showing_predictions:
            axs[0, 1].imshow(pred, **kwargs)
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
