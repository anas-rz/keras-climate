"""HySpecNet dataset (ported from torchgeo.datasets.hyspecnet).

.. note::
   This dataset depends on ``keras_climate.datasets.enmap.EnMAP``, which is
   ported by another file in this same porting effort. Importing this
   module will fail until that sibling module exists.
"""

import os
import re

import numpy as np
import rasterio as rio
from keras import ops

from .enmap import EnMAP
from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import (
    disambiguate_timestamp,
    download_url,
    extract_archive,
    quantile_normalization,
)


class HySpecNet11k(NonGeoDataset):
    """HySpecNet-11k dataset.

    `HySpecNet-11k <https://hyspecnet.rsim.berlin/>`__ is a large-scale
    benchmark dataset for hyperspectral image compression and
    self-supervised learning. It is made up of 11,483 nonoverlapping image
    patches acquired by the `EnMAP satellite <https://www.enmap.org/>`_.
    Each patch is a portion of 128 x 128 pixels with 224 spectral bands and
    with a ground sample distance of 30 m.

    We provide predefined splits obtained by randomly dividing HySpecNet
    into:

    #. a training set that includes 70% of the patches,
    #. a validation set that includes 20% of the patches, and
    #. a test set that includes 10% of the patches.

    Depending on the way that we used for splitting the dataset, we define
    two different splits:

    #. an easy split, where patches from the same tile can be present in
       different sets (patchwise splitting); and
    #. a hard split, where all patches from one tile belong to the same set
       (tilewise splitting).

    If you use this dataset in your research, please cite the following
    paper:

    * https://arxiv.org/abs/2306.00385
    """

    url = "https://hf.co/datasets/torchgeo/hyspecnet/resolve/13e110422a6925cbac0f11edff610219b9399227/"
    sha256s = {
        "hyspecnet-11k-01.tar.gz": "bd551f2b16d02ec6154b83dd75bc8640fdc4ef75b03680610d5cfec07d2e4d1d",
        "hyspecnet-11k-02.tar.gz": "245afe0be3b6bd5ac8783c6351390ffb16c1c2320db48f09adbcb4bedbf79844",
        "hyspecnet-11k-03.tar.gz": "31223d9b7ce1ebeb51138ab885658c0aeefa908df9fee4c2ca35605fc8a8c4ac",
        "hyspecnet-11k-04.tar.gz": "9283deea3188705d6b3d4a5239e792c43d30498cc8c7615eb859b55b72e1084e",
        "hyspecnet-11k-05.tar.gz": "c5eb190d0336fe459273cff5e87c597430771f273f0a5226388e7efe6fd0c263",
        "hyspecnet-11k-06.tar.gz": "8f9af9fcb4c22cc86066a6c60f876a65f4dc114d4c4be8e1d997e6a701c1ae6e",
        "hyspecnet-11k-07.tar.gz": "39a8b75e56451a5590a3ae378d14f15edfb3326bbad5dd664cc2ca07cf4edeb1",
        "hyspecnet-11k-08.tar.gz": "c45d379b047d0e685e465a534550e1f144e83303a98d980493858d5513add656",
        "hyspecnet-11k-09.tar.gz": "01d70d41f37f70e52ad562c5d7b3232784c8f3ec0ac8fdac9f7f4e44a7e8a7f8",
        "hyspecnet-11k-10.tar.gz": "643248dd6d0e9c5bd297c040e16aaa67aacc6e7c2162824009aea74b4e440bcf",
        "hyspecnet-11k-splits.tar.gz": "12d809d1e13ae4b2cf76b9eca2855e0bbab143372ebe728440eab383364cfb8b",
    }

    all_bands = EnMAP.all_bands
    default_bands = EnMAP.default_bands
    rgb_bands = EnMAP.rgb_bands

    def __init__(
        self,
        root="data",
        split="train",
        strategy="easy",
        bands=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new HySpecNet11k instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            strategy: Either 'easy' for patchwise splitting or 'hard' for
                tilewise splitting.
            bands: Bands to return.
            transforms: A function/transform that takes input sample and its
                target as entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, verify the checksum of the downloaded files.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.split = split
        self.strategy = strategy
        self.bands = bands or self.default_bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self.wavelengths = ops.convert_to_tensor(
            np.array([EnMAP.wavelengths[b] for b in self.bands], dtype="float32")
        )
        self.band_indices = [self.all_bands.index(b) + 1 for b in self.bands]

        self._verify()

        path = os.path.join(root, "hyspecnet-11k", "splits", strategy, f"{split}.csv")
        with open(path) as f:
            self.files = f.read().strip().split("\n")

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def __getitem__(self, index):
        """Return an index within the dataset."""
        path = self.files[index].replace("DATA.npy", "SPECTRAL_IMAGE.TIF")
        file = os.path.basename(path)
        match = re.match(EnMAP.filename_regex, file, re.VERBOSE)
        assert match
        mint, maxt = disambiguate_timestamp(match.group("date"), EnMAP.date_format)

        with rio.open(os.path.join(self.root, "hyspecnet-11k", "patches", path)) as src:
            minx, maxx = src.bounds.left, src.bounds.right
            miny, maxy = src.bounds.bottom, src.bounds.top
            array = src.read(self.band_indices).astype("float32")
            array = np.transpose(array, (1, 2, 0))
            sample = {
                "image": ops.convert_to_tensor(array),
                "x": ops.convert_to_tensor(np.array((minx + maxx) / 2, dtype="float64")),
                "y": ops.convert_to_tensor(np.array((miny + maxy) / 2, dtype="float64")),
                "t": ops.convert_to_tensor(
                    np.array(
                        (mint.timestamp() + maxt.timestamp()) / 2, dtype="float64"
                    )
                ),
                "wavelength": self.wavelengths,
                "res": ops.convert_to_tensor(np.array(30)),
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        exists = []
        for directory in ["patches", "splits"]:
            path = os.path.join(self.root, "hyspecnet-11k", directory)
            exists.append(os.path.isdir(path))

        if all(exists):
            return

        for file, sha256 in self.sha256s.items():
            # Check if the file has already been downloaded
            path = os.path.join(self.root, file)
            if os.path.isfile(path):
                extract_archive(path)
                continue

            # Check if the user requested to download the dataset
            if self.download:
                url = self.url + file
                download_url(url, self.root, sha256=sha256 if self.checksum else None)
                extract_archive(path)
                continue

            raise DatasetNotFoundError(self)

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

        image = ops.take(sample["image"], rgb_indices, axis=-1)
        image = quantile_normalization(image)
        image = ops.convert_to_numpy(image)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle:
            fig.suptitle(suptitle)

        return fig
