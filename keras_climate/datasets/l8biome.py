"""L8 Biome dataset (ported from torchgeo.datasets.l8biome)."""

import glob
import os

import numpy as np
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import IntersectionDataset, RasterDataset
from .utils import download_url, extract_archive, quantile_normalization


class L8BiomeImage(RasterDataset):
    """Images from the L8 Biome dataset."""

    # https://gisgeography.com/landsat-file-naming-convention/
    filename_glob = "LC8*.TIF"
    filename_regex = r"""
        ^LC8
        (?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        (?P<date>\d{7})
        (?P<gsi>[A-Z]{3})
        (?P<version>\d{2})
        \.TIF$
    """
    date_format = "%Y%j"
    is_image = True
    rgb_bands = ("B4", "B3", "B2")
    all_bands = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11")


class L8BiomeMask(RasterDataset):
    """Masks from the L8 Biome dataset."""

    # https://gisgeography.com/landsat-file-naming-convention/
    filename_glob = "LC8*_fixedmask.TIF"
    filename_regex = r"""
        ^LC8
        (?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        (?P<date>\d{7})
        (?P<gsi>[A-Z]{3})
        (?P<version>\d{2})
        _fixedmask
        \.TIF$
    """
    date_format = "%Y%j"
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


class L8Biome(IntersectionDataset):
    """L8 Biome dataset.

    The `L8 Biome <https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data>`__
    dataset is a validation dataset for cloud cover assessment algorithms,
    consisting of Pre-Collection Landsat 8 Operational Land Imager (OLI)
    Thermal Infrared Sensor (TIRS) terrain-corrected (Level-1T) scenes.

    Dataset features:

    * Images evenly divided between 8 unique biomes
    * 96 scenes from Landsat 8 OLI/TIRS sensors
    * Imagery from global tiles between April 2013--October 2014
    * 11 Level-1 spectral bands with 30 m per pixel resolution

    Dataset classes:

    0. Fill
    1. Cloud Shadow
    2. Clear
    3. Thin Cloud
    4. Cloud

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.5066/F7251GDH
    * https://doi.org/10.1016/j.rse.2017.03.026
    """

    url = "https://hf.co/datasets/torchgeo/l8biome/resolve/f76df19accce34d2acc1878d88b9491bc81f94c8/{}.tar.gz"

    sha256s = {
        "barren": "96104e03b50e6fed6a7e0d695133653c2b30893b6c25b4e06fe2f0947a2c96ea",
        "forest": "55b3728ae82a2cc67dec25cd0a262eb1f3e78f65d101b9f9a3c06577c851c912",
        "grass_crops": "86a436d51bb511022352b029f8acc8f9a308f390982cbbd3ae5024dd23ebf24c",
        "shrubland": "5a4891ebe03c45e97407e59b8101b0467317235015fc3ab12ca32c7e915a6be9",
        "snow_ice": "b8ef5ce0b3442a7444b48b6b4076bf975567d6d7764fc1c5bd60700936c0eb3f",
        "urban": "19e05ab8d80082f0ea6a6c53ab9989d6c7b5cbc93a3c18deebda8ca56bc43e9d",
        "water": "d9b6d554b1c8f7987d53ab81fa8fbd565bb73a87a702cc3cb161bf6a9373f83a",
        "wetlands": "ae8ad56992ca314ba6c10ba5e360f77b9590bb1d7b455032e5937bc078ffc8cc",
    }

    def __init__(
        self,
        paths,
        crs=None,
        res=None,
        bands=L8BiomeImage.all_bands,
        transforms=None,
        cache=True,
        download=False,
        checksum=True,
        time_series=False,
    ):
        """Initialize a new L8Biome instance.

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

        self.image = L8BiomeImage(
            paths, crs, res, bands, transforms, cache, time_series
        )
        self.mask = L8BiomeMask(paths, crs, res, None, transforms, cache, time_series)

        super().__init__(self.image, self.mask)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        if not isinstance(self.paths, (str, os.PathLike)):
            return

        paths = self.paths

        for classname in [L8BiomeImage, L8BiomeMask]:
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
