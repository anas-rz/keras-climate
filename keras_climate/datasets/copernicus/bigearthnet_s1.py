"""Copernicus-Bench BigEarthNet-S1 dataset (ported from torchgeo.datasets.copernicus.bigearthnet_s1)."""

import os

import numpy as np
import pandas as pd
from keras import ops

from .base import CopernicusBenchBase


class CopernicusBenchBigEarthNetS1(CopernicusBenchBase):
    """Copernicus-Bench BigEarthNet-S1 dataset.

    BigEarthNet-S1 is a multilabel land use/land cover classification dataset
    composed of 5% of the Sentinel-1 data of BigEarthNet-v2.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://arxiv.org/abs/2407.03653
    """

    url = "https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_bigearthnet_s1s2/bigearthnetv2.zip"
    sha256 = "f0d0444ce88b6d208d6cc29d3790c500eb1ceb9c188560bf7ab54f11ed99b31b"
    zipfile = "bigearthnetv2.zip"
    directory = "bigearthnet_s1s2"
    filename = "multilabel-{}.csv"
    filename_regex = r".{16}_(?P<date>\d{8}T\d{6})"
    all_bands = ("VV", "VH")
    rgb_bands = ("VV", "VH")
    classes = (
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    )

    def __init__(
        self,
        root="data",
        split="train",
        bands=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new CopernicusBenchBigEarthNetS1 instance.

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
        super().__init__(root, split, bands, transforms, download, checksum)
        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath)

    def __getitem__(self, index):
        row = self.files.iloc[index].values
        file = row[0]
        path = os.path.join(self.root, self.directory, "BigEarthNet-S1-5%", file)
        sample = self._load_image(path)
        sample["label"] = ops.convert_to_tensor(row[2:].astype(np.int64), dtype="int64")

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
