"""Copernicus-Bench LCZ-S2 dataset (ported from torchgeo.datasets.copernicus.lcz_s2)."""

import os

import numpy as np
from keras import ops

from ..errors import DatasetNotFoundError
from ..utils import download_url, lazy_import
from .base import CopernicusBenchBase


class CopernicusBenchLCZS2(CopernicusBenchBase):
    """Copernicus-Bench LCZ-S2 dataset.

    LCZ-S2 is a multi-class scene classification dataset derived from
    So2Sat-LCZ42, a large-scale local climate zone classification dataset.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://doi.org/10.1109/MGRS.2020.2964708

    .. note::

       This dataset requires the following additional library to be
       installed:

       * `<https://pypi.org/project/h5py/>`_ to load the dataset.
    """

    url = "https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l3_lcz_s2/lcz_{}.h5"
    sha256s = {
        "train": "b3e7297df254a959a00be63a312b81d6938bd201b4e8dbf77b5f9d0c9a48c54d",
        "val": "3d1a0d8aa4bd80a791b0ad8cfeeec2bbe21278218d294487f3d7eca8d63f4b4b",
        "test": "d2c923ed694ea4c531a9996ae924f8360c439cd7f9b5797fcc244f719e8a058f",
    }
    filename = "lcz_{}.h5"
    all_bands = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
    rgb_bands = ("B04", "B03", "B02")
    classes = (
        "Compact high rise",
        "Compact mid rise",
        "Compact low rise",
        "Open high rise",
        "Open mid rise",
        "Open low rise",
        "Lightweight low rise",
        "Large low rise",
        "Sparsely built",
        "Heavy industry",
        "Dense trees",
        "Scattered trees",
        "Bush, scrub",
        "Low plants",
        "Bare rock or paved",
        "Bare soil or sand",
        "Water",
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
        """Initialize a new CopernicusBenchLCZS2 instance.

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
        h5py = lazy_import("h5py")

        self.root = root
        self.split = split
        self.bands = bands or self.all_bands
        self.band_indices = [self.all_bands.index(i) for i in self.bands]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.filepath = os.path.join(root, self.filename.format(split))
        with h5py.File(self.filepath, "r") as f:
            self.length = f["label"].shape[0]

    def __len__(self):
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, index):
        """Return an index within the dataset."""
        h5py = lazy_import("h5py")

        with h5py.File(self.filepath, "r") as f:
            # Already stored channels-last (H x W x C)
            sen2 = f["sen2"][index][:, :, self.band_indices]
            label = f["label"][index].argmax()

        sample = {
            "image": ops.convert_to_tensor(sen2.astype("float32")),
            "label": ops.convert_to_tensor(np.array(label, dtype="int64")),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.filename.format(self.split))):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()

    def _download(self):
        """Download the dataset."""
        sha256 = self.sha256s[self.split] if self.checksum else None
        download_url(self.url.format(self.split), self.root, sha256=sha256)
