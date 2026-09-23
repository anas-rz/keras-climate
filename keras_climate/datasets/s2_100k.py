"""S2-100k pre-training dataset from the SatCLIP paper (ported from torchgeo.datasets.s2_100k).

Adapted from https://github.com/microsoft/satclip (Copyright (c) Microsoft Corporation).
"""

import pathlib

import numpy as np
import pandas as pd
import rasterio as rio
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import array_to_tensor, download_and_extract_archive, download_url, extract_archive


class S2100k(NonGeoDataset):
    """S2-100K dataset.

    The `S2-100k dataset <https://hf.co/datasets/kklmmr/s2-100k>`__
    contains 100,000 256x256 patches of 12 band Sentinel imagery sampled
    randomly from Sentinel 2 scenes on the Microsoft Planetary Computer that
    have <20% cloud cover, intersect land, and were captured between
    2021-01-01 and 2023-05-17.

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1609/aaai.v39i4.32457
    """

    url = "https://hf.co/datasets/torchgeo/s2-100k/resolve/fbdbb78ba57d22d5b6be203913f1e6020c2b4e0a"
    index_sha256 = "9fdcdec776b331fcc2d9ab5af18355efc5bd0716df33ab78e1ff03f60cf343ad"
    data_sha256 = "da1bae4e9dd44fb00e5f1fc537b752f5025ac908a73b5a9e24ff90bcbdd56edb"

    def __init__(
        self, root="data", mode="both", transforms=None, download=False, checksum=True
    ):
        """Initialize a new S2100K dataset instance.

        Args:
            root: root directory where dataset can be found
            mode: which data to return (options are "both" or "points"),
                useful for embedding locations without loading images
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: If *mode* argument is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert mode in {"both", "points"}

        self.root = pathlib.Path(root)
        self.transforms = transforms
        self.mode = mode
        self.download = download
        self.checksum = checksum

        self._verify()

        self.index = pd.read_csv(self.root / "index.csv")

    def __getitem__(self, index):
        row = self.index.iloc[index]

        point = ops.convert_to_tensor(
            np.array([row["lon"], row["lat"]], dtype="float32")
        )
        sample = {"point": point}

        if self.mode == "both":
            with rio.open(self.root / "images" / row["fn"]) as f:
                array = np.transpose(f.read(), (1, 2, 0))
                sample["image"] = ops.cast(array_to_tensor(array), "float32")

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.index)

    def _verify(self):
        """Verify the integrity of the dataset."""
        filename = "index.csv"
        if (self.root / filename).is_file():
            pass
        elif self.download:
            url = f"{self.url}/{filename}"
            sha256 = self.index_sha256 if self.checksum else None
            download_url(url, self.root, sha256=sha256)
        else:
            raise DatasetNotFoundError(self)

        if (self.root / "images" / "patch_0.tif").is_file():
            return

        path = self.root / "satclip.tar"
        if path.is_file():
            extract_archive(path, self.root / "images")
        elif self.download:
            url = f"{self.url}/satclip.tar"
            sha256 = self.data_sha256 if self.checksum else None
            download_and_extract_archive(url, self.root / "images", sha256=sha256)
        else:
            raise DatasetNotFoundError(self)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        image = np.take(image, [3, 2, 1], axis=-1)
        image = np.clip(image / 4000, 0, 1)

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            point = ops.convert_to_numpy(sample["point"])
            ax.set_title(f"({point[0]:0.4f}, {point[1]:0.4f})")

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
