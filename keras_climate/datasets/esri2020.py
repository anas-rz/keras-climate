"""Esri 2020 Land Cover Dataset (ported from torchgeo.datasets.esri2020)."""

import glob
import os

from keras import ops

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import download_url, extract_archive


class Esri2020(RasterDataset):
    """Esri 2020 Land Cover Dataset.

    The `Esri 2020 Land Cover dataset
    <https://www.arcgis.com/home/item.html?id=fc92d38533d440078f17678ebc20e8e2>`_
    consists of a global single band land use/land cover map derived from ESA
    Sentinel-2 imagery at 10m resolution with a total of 10 classes. It was
    published in July 2021 and used the Universal Transverse Mercator (UTM)
    projection. This dataset only contains labels, no raw satellite imagery.

    The 10 classes are:

    0. No Data
    1. Water
    2. Trees
    3. Grass
    4. Flooded Vegetation
    5. Crops
    6. Scrub/Shrub
    7. Built Area
    8. Bare Ground
    9. Snow/Ice
    10. Clouds

    If you use this dataset please cite the following paper:

    * https://ieeexplore.ieee.org/document/9553499
    """

    is_image = False
    filename_glob = "*_20200101-20210101.*"
    filename_regex = r"""^
        (?P<id>[0-9][0-9][A-Z])
        _(?P<date>\d{8})
        -(?P<processing_date>\d{8})
    """

    zipfile = "io-lulc-model-001-v01-composite-v03-supercell-v02-clip-v01.zip"
    md5 = "4932855fcd00735a34b74b1f87db3df0"

    url = (
        "https://ai4edataeuwest.blob.core.windows.net/io-lulc/"
        "io-lulc-model-001-v01-composite-v03-supercell-v02-clip-v01.zip"
    )

    def __init__(
        self,
        paths="data",
        crs=None,
        res=None,
        transforms=None,
        cache=True,
        download=False,
        checksum=True,
        time_series=False,
    ):
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS (defaults to the
                resolution of the first file found)
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)
            time_series: if True, stack data along the time series dimension

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

    def _verify(self):
        """Verify the integrity of the dataset."""
        if self.files:
            return

        assert isinstance(self.paths, (str, os.PathLike))
        pathname = os.path.join(self.paths, self.zipfile)
        if glob.glob(pathname):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self):
        """Download the dataset."""
        download_url(self.url, self.paths, filename=self.zipfile, md5=self.md5)

    def _extract(self):
        """Extract the dataset."""
        extract_archive(os.path.join(self.paths, self.zipfile))

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        mask = ops.convert_to_numpy(sample["mask"]).squeeze()
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = ops.convert_to_numpy(sample["prediction"]).squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(prediction)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
