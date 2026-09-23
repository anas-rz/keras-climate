"""Copernicus-Bench Biomass-S3 dataset (ported from torchgeo.datasets.copernicus.biomass_s3)."""

import glob
import os

import pandas as pd
from keras import ops

from ..utils import stack_samples
from .base import CopernicusBenchBase


class CopernicusBenchBiomassS3(CopernicusBenchBase):
    """Copernicus-Bench Biomass-S3 dataset.

    Biomass-S3 is a regression dataset based on Sentinel-3 OLCI images and CCI biomass.
    The biomass product is part of the European Space Agency's Climate Change Initiative
    (CCI) program and delivers global forest above-ground biomass at 100 m spatial
    resolution.

    This benchmark supports both static (1 image/location) and time series
    (1-4 images/location) modes, the former is used in the original benchmark.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://catalogue.ceda.ac.uk/uuid/02e1b18071ad45a19b4d3e8adafa2817/
    """

    url = "https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l3_biomass_s3/biomass_s3.zip"
    sha256 = "1d005b200d50f2e8b5f4482959bdfa6e2d7d05a8cd828d7f438c99a4e1cfbaef"
    zipfile = "biomass_s3.zip"
    directory = "biomass_s3"
    filename = "static_fnames-{}.csv"
    dtype = "float32"
    filename_regex = r"S3[AB]_(?P<date>\d{8}T\d{6})"
    all_bands = (
        "Oa01_radiance",
        "Oa02_radiance",
        "Oa03_radiance",
        "Oa04_radiance",
        "Oa05_radiance",
        "Oa06_radiance",
        "Oa07_radiance",
        "Oa08_radiance",
        "Oa09_radiance",
        "Oa10_radiance",
        "Oa11_radiance",
        "Oa12_radiance",
        "Oa13_radiance",
        "Oa14_radiance",
        "Oa15_radiance",
        "Oa16_radiance",
        "Oa17_radiance",
        "Oa18_radiance",
        "Oa19_radiance",
        "Oa20_radiance",
        "Oa21_radiance",
    )
    rgb_bands = ("Oa08_radiance", "Oa06_radiance", "Oa04_radiance")
    cmap = "YlGn"
    image_size = (282, 282)

    def __init__(
        self,
        root="data",
        split="train",
        mode="static",
        bands=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new CopernicusBenchBiomassS3 instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: One of 'static' or 'time-series'.
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
        self.mode = mode
        super().__init__(root, split, bands, transforms, download, checksum)
        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath, header=None)

    def _load_image(self, path):
        """Load and resize an image, replacing its declared nodata value."""
        sample = super()._load_image(path)
        image = sample["image"]
        image = ops.where(ops.isneginf(image), ops.zeros_like(image), image)
        image = ops.image.resize(image, self.image_size, interpolation="bilinear")
        sample["image"] = image
        return sample

    def _load_mask(self, path):
        """Load and resize a biomass mask."""
        sample = super()._load_mask(path)
        mask = ops.expand_dims(sample["mask"], axis=-1)
        mask = ops.image.resize(mask, self.image_size, interpolation="bilinear")
        sample["mask"] = ops.squeeze(mask, axis=-1)
        return sample

    def __getitem__(self, index):
        pid, file = self.files.iloc[index]
        if self.mode == "static":
            path = os.path.join(self.root, self.directory, "s3_olci", pid, file)
            sample = self._load_image(path)
        else:
            paths = os.path.join(self.root, self.directory, "s3_olci", pid, "*.tif")
            samples = [self._load_image(path) for path in sorted(glob.glob(paths))]
            sample = stack_samples(samples)

        path = os.path.join(self.root, self.directory, "biomass", f"{pid}.tif")
        sample |= self._load_mask(path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
