"""LandCover.ai dataset (ported from torchgeo.datasets.landcoverai)."""

import abc
import glob
import os

import numpy as np
import pandas as pd
import rasterio
from keras import ops
from matplotlib.colors import ListedColormap
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset, RasterDataset
from .utils import download_url, extract_archive


class LandCoverAIBase(abc.ABC):
    """Abstract base class for LandCover.ai Geo and NonGeo datasets.

    The `LandCover.ai <https://landcover.ai.linuxpolska.com/>`__ (Land Cover from
    Aerial Imagery) dataset is a dataset for automatic mapping of buildings,
    woodlands, water and roads from aerial images. This implementation is
    specifically for Version 1 of LandCover.ai.

    Dataset classes:

    1. background
    2. building
    3. woodland
    4. water
    5. road

    If you use this dataset in your research, please cite:

    * https://arxiv.org/abs/2005.02264v4
    """

    url = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
    filename = "landcover.ai.v1.zip"
    sha256 = "dd448287ff3a76291f2b159084b8a265f2ba4559bb80b0551fc8c6c0eb149a4b"
    classes = ("Background", "Building", "Woodland", "Water", "Road")
    cmap = ListedColormap(
        np.array(
            [
                (0, 0, 0, 0),
                (97, 74, 74, 255),
                (38, 115, 0, 255),
                (0, 197, 255, 255),
                (207, 207, 207, 255),
            ]
        )
        / 255
    )

    def __init__(self, root="data", download=False, checksum=True):
        """Initialize a new LandCover.ai dataset instance.

        Args:
            root: root directory where dataset can be found
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

    def _verify(self):
        """Verify the integrity of the dataset."""
        if self._verify_data():
            return

        pathname = os.path.join(self.root, self.filename)
        if os.path.exists(pathname):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    @abc.abstractmethod
    def _verify_data(self):
        """Verify if the images and masks are present."""

    def _download(self):
        """Download the dataset."""
        download_url(self.url, self.root, sha256=self.sha256 if self.checksum else None)

    def _extract(self):
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.filename))

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"]).astype("uint8").squeeze()
        mask = ops.convert_to_numpy(sample["mask"]).astype("uint8").squeeze()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = ops.convert_to_numpy(sample["prediction"])
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=4, cmap=self.cmap, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(
                predictions, vmin=0, vmax=4, cmap=self.cmap, interpolation="none"
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class LandCoverAIGeo(LandCoverAIBase, RasterDataset):
    """LandCover.ai Geo dataset.

    See the abstract LandCoverAIBase class to find out more.
    """

    filename_glob = os.path.join("images", "*.tif")
    filename_regex = ".*tif"

    def __init__(
        self,
        root="data",
        crs=None,
        res=None,
        transforms=None,
        cache=True,
        download=False,
        checksum=True,
        time_series=False,
    ):
        """Initialize a new LandCover.ai Geo dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS (defaults to the
                resolution of the first file found)
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)
            time_series: if True, stack data along the time series dimension
                (``[T, H, W, C]``). If False, merge data into a
                (``[H, W, C]``) mosaic.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        LandCoverAIBase.__init__(self, root, download, checksum)
        RasterDataset.__init__(
            self,
            root,
            crs,
            res,
            transforms=transforms,
            cache=cache,
            time_series=time_series,
        )

    def _verify_data(self):
        """Verify if the images and masks are present."""
        img_query = os.path.join(self.root, "images", "*.tif")
        mask_query = os.path.join(self.root, "masks", "*.tif")
        images = glob.glob(img_query)
        masks = glob.glob(mask_query)
        return len(images) > 0 and len(images) == len(masks)

    def __getitem__(self, index):
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        img_filepaths = df.filepath
        mask_filepaths = img_filepaths.apply(lambda x: x.replace("images", "masks"))

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        img = self._merge_or_stack(img_filepaths, index, self.band_indexes)
        mask = self._merge_or_stack(mask_filepaths, index, self.band_indexes)
        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            "bounds": self._slice_to_tensor(index),
            "image": ops.cast(img, "float32"),
            "mask": ops.cast(ops.squeeze(mask, axis=-1), "int64"),
            "transform": ops.convert_to_tensor(np.array(list(transform))),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class LandCoverAI(LandCoverAIBase, NonGeoDataset):
    """LandCover.ai dataset.

    See the abstract LandCoverAIBase class to find out more.

    .. note::

       This dataset uses a pre-chipped version of the data available on
       HuggingFace. The pre-chipped dataset contains the output/ directory
       with 512x512 image chips and corresponding masks in JPG/PNG format.
    """

    url = "https://hf.co/datasets/dragon7/LandCover.ai/resolve/262c75fbbf77d107f0a8335e9eef1f6234481d08/output.zip"
    filename = "output.zip"
    sha256 = "0edd6b7049d089519e63450e371037890b5e9552b1df056dbbddcc88770fd0da"
    metadata = {
        "train": {
            "filename": "train.txt",
            "sha256": "3db55adca08bb2161448875e7798102a313a9bd4b7a2caf60892293f0ca98450",
        },
        "val": {
            "filename": "val.txt",
            "sha256": "521d28411921e97602eb619a55058ddb3702a743f9b4c3aefd56130ae122c6ea",
        },
        "test": {
            "filename": "test.txt",
            "sha256": "df381ab217cb7bfd526cca4249bc863cb11d8b99edbe6837b086f513fe9c123f",
        },
    }

    def __init__(
        self, root="data", split="train", transforms=None, download=False, checksum=True
    ):
        """Initialize a new LandCover.ai dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in ["train", "val", "test"]

        super().__init__(root, download, checksum)

        self.transforms = transforms
        self.split = split
        with open(os.path.join(self.root, split + ".txt")) as f:
            self.ids = f.readlines()

    def __getitem__(self, index):
        """Return an index within the dataset."""
        id_ = self.ids[index].rstrip()
        sample = {"image": self._load_image(id_), "mask": self._load_target(id_)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.ids)

    def _load_image(self, id_):
        """Load a single image."""
        filename = os.path.join(self.root, "output", id_ + ".jpg")
        with Image.open(filename) as img:
            array = np.array(img)
            tensor = ops.cast(ops.convert_to_tensor(array), "float32")
            return tensor

    def _load_target(self, id_):
        """Load the target mask for a single image."""
        filename = os.path.join(self.root, "output", id_ + "_m.png")
        with Image.open(filename) as img:
            array = np.array(img.convert("L"))
            tensor = ops.cast(ops.convert_to_tensor(array), "int64")
            return tensor

    def _verify_data(self):
        """Verify if the images and masks are present."""
        for split in ["train", "val", "test"]:
            if not os.path.exists(os.path.join(self.root, f"{split}.txt")):
                return False

        img_query = os.path.join(self.root, "output", "*_*.jpg")
        mask_query = os.path.join(self.root, "output", "*_*_m.png")
        images = glob.glob(img_query)
        masks = glob.glob(mask_query)
        return len(images) > 0 and len(images) == len(masks)

    def _download(self):
        """Download the dataset and split files."""
        download_url(self.url, self.root, sha256=self.sha256 if self.checksum else None)

        for split_info in self.metadata.values():
            split_url = self.url.replace(self.filename, split_info["filename"])
            download_url(
                split_url,
                self.root,
                sha256=split_info["sha256"] if self.checksum else None,
            )


class LandCoverAI100(LandCoverAI):
    """Subset of LandCoverAI containing only 100 images.

    Intended for tutorials and demonstrations, not for benchmarking.
    Maintains the same file structure, classes, and train-val-test split.
    """

    url = "https://hf.co/datasets/isaaccorley/landcoverai/resolve/5cdf9299bd6c1232506cf79373df01f6e6596b50/landcoverai100.zip"
    filename = "landcoverai100.zip"
    sha256 = "bfe5bcf501a54cfd8ebf985346da50be5e8b751d3491812cd0c226b5a3abff41"

    def _download(self):
        """Download the dataset.

        Unlike the parent LandCoverAI class, LandCoverAI100 includes split
        files in the zip, so we don't need to download them separately.
        """
        download_url(self.url, self.root, sha256=self.sha256 if self.checksum else None)
