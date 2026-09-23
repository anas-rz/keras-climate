"""L7 Irish dataset (ported from torchgeo.datasets.l7irish)."""

import glob
import os

import numpy as np
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import IntersectionDataset, RasterDataset
from .utils import download_url, extract_archive, quantile_normalization


class L7IrishImage(RasterDataset):
    """Images from the L7 Irish dataset."""

    # https://landsat.usgs.gov/cloud-validation/cca_irish_2015/L7_Irish_Cloud_Validation_Masks.xml
    filename_glob = "L71*.TIF"
    filename_regex = r"""
        ^L71
        (?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        _(?P=wrs_row)
        (?P<date>\d{8})
        \.TIF$
    """
    date_format = "%Y%m%d"
    is_image = True
    rgb_bands = ("B30", "B20", "B10")
    all_bands = ("B10", "B20", "B30", "B40", "B50", "B61", "B62", "B70", "B80")


class L7IrishMask(RasterDataset):
    """Masks from the L7 Irish dataset."""

    # https://landsat.usgs.gov/cloud-validation/cca_irish_2015/L7_Irish_Cloud_Validation_Masks.xml
    filename_glob = "L7_p*_r*_newmask2015.TIF"
    filename_regex = r"""
        ^L7
        _p(?P<wrs_path>\d+)
        _r(?P<wrs_row>\d+)
        _newmask2015\.TIF$
    """
    is_image = False
    classes = ("Fill", "Cloud Shadow", "Clear", "Thin Cloud", "Cloud")
    ordinal_map = np.zeros(256, dtype="int64")
    ordinal_map[64] = 1
    ordinal_map[128] = 2
    ordinal_map[192] = 3
    ordinal_map[255] = 4

    def __getitem__(self, index):
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        sample = super().__getitem__(index)
        mask = ops.convert_to_numpy(sample["mask"]).astype("int64")
        sample["mask"] = ops.convert_to_tensor(self.ordinal_map[mask])
        return sample


class L7Irish(IntersectionDataset):
    """L7 Irish dataset.

    The `L7 Irish <https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data>`__
    dataset is based on Landsat 7 Enhanced Thematic Mapper Plus (ETM+)
    Level-1G scenes. Manually generated cloud masks are used to train and
    validate cloud cover assessment algorithms, which in turn are intended
    to compute the percentage of cloud cover in each scene.

    Dataset features:

    * Images divided between 9 unique biomes
    * 206 scenes from Landsat 7 ETM+ sensor
    * Imagery from global tiles between June 2000--December 2001
    * 9 Level-1 spectral bands with 30 m per pixel resolution

    Dataset classes:

    0. Fill
    1. Cloud Shadow
    2. Clear
    3. Thin Cloud
    4. Cloud

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.5066/F7XD0ZWC
    * https://doi.org/10.1109/TGRS.2011.2164087
    * https://www.sciencebase.gov/catalog/item/573ccf18e4b0dae0d5e4b109
    """

    url = "https://hf.co/datasets/torchgeo/l7irish/resolve/6807e0b22eca7f9a8a3903ea673b31a115837464/{}.tar.gz"

    sha256s = {
        "austral": "9b025debb20791cd3279cbc56f39dcd42fa7f20f172608a750e06b31c153457e",
        "boreal": "7d5bf24420e7606b71669c39e7ffc7fbef0605224845e6c6995572d3fadffff2",
        "mid_latitude_north": "7ab40faee550f941da41365093cf604c304641343c9777bbe9dba46d050e4a4f",
        "mid_latitude_south": "8b74d7debd5229d03fe210c6ce813a7e4a8b2ece7acc3a8f8d2af8100e6e034d",
        "polar_north": "0eb82d2c5a46600b7d4ffe1e67e4f0858947707e183dc29c00027b3f7caae3d1",
        "polar_south": "1d1b89e232af2d2685355713c1a11eb8de351ca7dc8e34d82b87a6a4237d47f4",
        "subtropical_north": "48f5cbd08b6095ae853f632e5660ade0d99be5c713d5a05b49302fbe5070860d",
        "subtropical_south": "ad88b32b992fcf88aab9c7e83c678c8e71f75547bdf7d66baec00aafdb0fdcad",
        "tropical": "659c5f528b81f9e8626a3b88ce47b844484522a3316cbc62be7a0cdfd994a7b4",
    }

    def __init__(
        self,
        paths="data",
        crs=None,
        res=None,
        bands=L7IrishImage.all_bands,
        transforms=None,
        cache=True,
        download=False,
        checksum=True,
        time_series=False,
    ):
        """Initialize a new L7Irish instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to EPSG:3857)
            res: resolution of the dataset in units of CRS
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum of the downloaded files
            time_series: if True, stack data along the time series dimension

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        from pyproj import CRS as PROJ_CRS

        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        if crs is None:
            crs = PROJ_CRS.from_epsg(3857)

        self.image = L7IrishImage(
            paths, crs, res, bands, transforms, cache, time_series
        )
        self.mask = L7IrishMask(paths, crs, res, None, transforms, cache, time_series)

        # Mask filename does not include the date, grab it from the image filename
        self.mask.index.index = self.image.index.index

        super().__init__(self.image, self.mask)

        # Ignore unintentional partial overlap
        self.index = self.image.index

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        if not isinstance(self.paths, (str, os.PathLike)):
            return

        paths = self.paths

        for classname in [L7IrishImage, L7IrishMask]:
            pathname = os.path.join(paths, "**", classname.filename_glob)
            if not glob.glob(pathname, recursive=True):
                break
        else:
            return

        # Check if the tar.gz files have already been downloaded
        pathname = os.path.join(paths, "*.tar.gz")
        if glob.glob(pathname):
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
        paths = self.paths
        for biome, sha256 in self.sha256s.items():
            download_url(
                self.url.format(biome), paths, sha256=sha256 if self.checksum else None
            )

    def _extract(self):
        """Extract the dataset."""
        paths = self.paths
        pathname = os.path.join(paths, "*.tar.gz")
        for tarfile in glob.iglob(pathname):
            extract_archive(tarfile)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        import matplotlib.pyplot as plt

        rgb_indices = []
        for band in self.image.rgb_bands:
            if band in self.image.bands:
                rgb_indices.append(self.image.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = ops.take(sample["image"], rgb_indices, axis=-1)
        image = quantile_normalization(image)
        image = ops.convert_to_numpy(image)

        mask = ops.convert_to_numpy(sample["mask"]).astype("uint8").squeeze()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = (
                ops.convert_to_numpy(sample["prediction"]).astype("uint8").squeeze()
            )
            num_panels += 1

        kwargs = {"cmap": "gray", "vmin": 0, "vmax": 4, "interpolation": "none"}
        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, **kwargs)
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions, **kwargs)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
