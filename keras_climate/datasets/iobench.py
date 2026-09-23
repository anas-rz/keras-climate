"""I/O benchmark dataset (ported from torchgeo.datasets.iobench).

.. note::
   This dataset depends on ``keras_climate.datasets.cdl.CDL`` and
   ``keras_climate.datasets.landsat.Landsat9``, which are ported by other
   files in this same porting effort. Importing this module will fail until
   those sibling modules exist.
"""

import glob
import os

from keras import ops

from .cdl import CDL
from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import IntersectionDataset
from .landsat import Landsat9
from .utils import download_url, extract_archive, quantile_normalization


class IOBench(IntersectionDataset):
    """I/O Bench dataset.

    I/O Bench is a dataset designed to benchmark the I/O performance of
    keras_climate/torchgeo. It contains a single Landsat 9 scene and CDL
    file from 2023, and consists of the following splits

    * original: the original files as downloaded
      from USGS Earth Explorer and USDA CropScape
    * raw: the same files with compression and with
      CDL clipped to the bounds of the Landsat scene
    * preprocessed: the same files with compression,
      reprojected to the same CRS, as COGs, with TAP

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.1145/3557915.3560953
    """

    url = "https://hf.co/datasets/torchgeo/io/resolve/c9d9d268cf0b61335941bdc2b6963bf16fc3a6cf/{}.tar.gz"

    sha256s = {
        "original": "a346f1ce4331ba0b75f1554a34d0ff0635732e59102f2ad9fdb0b28b00a74dc3",
        "raw": "af3fd271b98fd88abbca30bf43b07718b40d32eb40eff0eebf6446da11f43086",
        "preprocessed": "ab4eaa5f48cda9e6bf74f77b266990c787c45c2ec5dacb4889d90c9a3fbc9a9d",
    }

    def __init__(
        self,
        root="data",
        split="preprocessed",
        crs=None,
        res=None,
        bands=None,
        classes=None,
        transforms=None,
        cache=True,
        download=False,
        checksum=True,
    ):
        """Initialize a new IOBench instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'original', 'raw', or 'preprocessed'.
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: Resolution of the dataset in units of CRS.
            bands: Bands to return (defaults to all bands).
            classes: List of classes to include, the rest will be mapped to 0.
            transforms: A function/transform that takes an input sample
                and returns a transformed version.
            cache: If True, cache file handle to speed up repeated sampling.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, verify the checksum of the downloaded files.

        Raises:
            AssertionError: If *split* argument is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        if bands is None:
            bands = [*Landsat9.default_bands, "SR_QA_AEROSOL"]
        if classes is None:
            classes = list(CDL.valid_classes)
        assert split in self.sha256s

        self.root = root
        self.split = split
        self.download = download
        self.checksum = checksum

        self._verify()

        root = os.path.join(root, split)
        self.landsat = Landsat9(root, crs, res, bands, transforms, cache)
        self.cdl = CDL(root, crs, res, [2023], classes, transforms, cache)

        super().__init__(self.landsat, self.cdl)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        count = 0
        for filename_glob in [Landsat9.filename_glob[:6], CDL.filename_glob]:
            pathname = os.path.join(self.root, self.split, "**", filename_glob)
            count += len(glob.glob(pathname, recursive=True))

        if count == 9:
            return

        # Check if the tar files have already been downloaded
        if glob.glob(os.path.join(self.root, f"{self.split}.tar.gz")):
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
            self.url.format(self.split),
            self.root,
            sha256=self.sha256s[self.split] if self.checksum else None,
        )

    def _extract(self):
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, f"{self.split}.tar.gz"), self.root)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        import matplotlib.pyplot as plt

        rgb_indices = []
        for band in self.landsat.rgb_bands:
            if band in self.landsat.bands:
                rgb_indices.append(self.landsat.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = ops.take(sample["image"], rgb_indices, axis=-1)
        image = ops.cast(image, "float32")
        image = quantile_normalization(image)
        image = ops.convert_to_numpy(image)

        mask = ops.convert_to_numpy(ops.squeeze(sample["mask"])).astype("int64")
        mask = self.cdl.inverse_map[mask]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        kwargs = {"cmap": self.cdl.cmap, "vmin": 0, "vmax": 255, "interpolation": "none"}

        axes[0].imshow(image)
        axes[1].imshow(mask, **kwargs)

        axes[0].axis("off")
        axes[1].axis("off")

        if show_titles:
            axes[0].set_title("Image")
            axes[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
