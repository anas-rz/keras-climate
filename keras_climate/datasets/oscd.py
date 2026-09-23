"""OSCD dataset (ported from torchgeo.datasets.oscd)."""

import glob
import os
import warnings

import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import (
    download_url,
    extract_archive,
    quantile_normalization,
    sort_sentinel2_bands,
)


class OSCD(NonGeoDataset):
    """OSCD dataset.

    The `Onera Satellite Change Detection <https://rcdaudt.github.io/oscd/>`_
    dataset addresses the issue of detecting changes between satellite
    images from different dates. Imagery comes from Sentinel-2 which
    contains varying resolutions per band.

    Dataset format:

    * images are 13-channel tifs
    * masks are single-channel pngs where no change = 0, change = 255

    Dataset classes:

    0. no change
    1. change

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.1109/IGARSS.2018.8518015
    """

    urls = {
        "Onera Satellite Change Detection dataset - Images.zip": "https://hf.co/datasets/hkristen/oscd/resolve/4958d786c1389ede1511d91a6ecf1a75c4074933/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images.zip",
        "Onera Satellite Change Detection dataset - Train Labels.zip": "https://hf.co/datasets/hkristen/oscd/resolve/4958d786c1389ede1511d91a6ecf1a75c4074933/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Train%20Labels.zip",
        "Onera Satellite Change Detection dataset - Test Labels.zip": "https://hf.co/datasets/hkristen/oscd/resolve/4958d786c1389ede1511d91a6ecf1a75c4074933/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Test%20Labels.zip",
    }
    sha256s = {
        "Onera Satellite Change Detection dataset - Images.zip": (
            "940b87887511058a933e67cd6d0e43e2eb825a55d8e79a50983dee7f23003656"
        ),
        "Onera Satellite Change Detection dataset - Train Labels.zip": (
            "89fb54cd12ad0dbea6c447528139dec305b865294215434bf6dd170fb8fd3ca5"
        ),
        "Onera Satellite Change Detection dataset - Test Labels.zip": (
            "2e195eaa1b788b99fa93ea8073e3780bc0b763000b0c49dbf70548acf1e5d67d"
        ),
    }

    zipfile_glob = "*Onera*.zip"
    filename_glob = "*Onera*"
    splits = ("train", "test")

    all_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    )

    rgb_bands = ("B04", "B03", "B02")

    def __init__(
        self,
        root="data",
        split="train",
        bands=all_bands,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new OSCD dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert split in self.splits
        assert set(bands) <= set(self.all_bands)
        self.bands = bands
        self.all_band_indices = [self.all_bands.index(b) for b in self.bands]

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

    def __getitem__(self, index):
        """Return an index within the dataset.

        .. versionchanged::
           Now returns a single T x H x W x C image.
        """
        files = self.files[index]
        image1 = self._load_image(files["images1"])
        image2 = self._load_image(files["images2"])
        mask = self._load_target(files["mask"])
        image = ops.stack([image1, image2], axis=0)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def _load_files(self):
        regions = []
        labels_root = os.path.join(
            self.root,
            f"Onera Satellite Change Detection dataset - {self.split.capitalize()} "
            + "Labels",
        )
        images_root = os.path.join(
            self.root, "Onera Satellite Change Detection dataset - Images"
        )
        folders = glob.glob(os.path.join(labels_root, "*/"))
        for folder in folders:
            region = folder.split(os.sep)[-2]
            mask = os.path.join(labels_root, region, "cm", "cm.png")

            def get_image_paths(ind, region=region):
                return sorted(
                    glob.glob(
                        os.path.join(images_root, region, f"imgs_{ind}_rect", "*.tif")
                    ),
                    key=sort_sentinel2_bands,
                )

            images1, images2 = get_image_paths(1), get_image_paths(2)
            images1 = [images1[i] for i in self.all_band_indices]
            images2 = [images2[i] for i in self.all_band_indices]

            with open(os.path.join(images_root, region, "dates.txt")) as f:
                dates = tuple(line.split()[-1] for line in f.read().strip().splitlines())

            regions.append(
                {
                    "region": region,
                    "images1": images1,
                    "images2": images2,
                    "mask": mask,
                    "dates": dates,
                }
            )

        return regions

    def _load_image(self, paths):
        """Load a single image."""
        images = []
        for path in paths:
            with Image.open(path) as img:
                images.append(np.array(img))
        # (C, H, W) -> (H, W, C)
        array = np.stack(images, axis=-1).astype("float32")
        return ops.convert_to_tensor(array)

    def _load_target(self, path):
        """Load the target mask for a single image."""
        with Image.open(path) as img:
            array = np.array(img.convert("L"))
            array = np.clip(array, 0, 1).astype("int64")
            # A time dimension is kept for consistency with the image tensor
            array = array[np.newaxis, ...]
            return ops.convert_to_tensor(array)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "**", self.filename_glob)
        for fname in glob.iglob(pathname, recursive=True):
            if not fname.endswith(".zip"):
                return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_glob)
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
        for filename in self.urls:
            download_url(
                self.urls[filename],
                self.root,
                filename=filename,
                sha256=self.sha256s[filename] if self.checksum else None,
            )

    def _extract(self):
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.zipfile_glob)
        for zipfile in glob.iglob(pathname):
            extract_archive(zipfile)

    def plot(self, sample, show_titles=True, suptitle=None, alpha=None):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        if alpha is not None:
            warnings.warn(
                "The alpha parameter is deprecated and has no effect.",
                DeprecationWarning,
                stacklevel=2,
            )

        import matplotlib.pyplot as plt

        try:
            rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        except ValueError as e:
            raise RGBBandsMissingError() from e

        def to_rgb(img):
            rgb = ops.take(img, np.array(rgb_indices), axis=-1)
            return quantile_normalization(rgb)

        ncols = 3
        if "prediction" in sample:
            ncols = 4

        image1 = ops.convert_to_numpy(to_rgb(sample["image"][0]))
        image2 = ops.convert_to_numpy(to_rgb(sample["image"][1]))
        mask = ops.convert_to_numpy(sample["mask"][0])

        h, w = image1.shape[:2]
        fig, axs = plt.subplots(
            1, ncols, figsize=(ncols * 5, 5 * h / w), layout="constrained"
        )
        axs[0].imshow(image1)
        axs[1].imshow(image2)
        axs[2].imshow(mask, cmap="gray", interpolation="none", vmin=0, vmax=1)
        if ncols == 4:
            axs[3].imshow(
                ops.convert_to_numpy(sample["prediction"][0]),
                cmap="gray",
                interpolation="none",
                vmin=0,
                vmax=1,
            )

        for ax in axs:
            ax.axis("off")

        if show_titles:
            axs[0].set_title("Pre-change (T1)")
            axs[1].set_title("Post-change (T2)")
            axs[2].set_title("Ground Truth")
            if ncols == 4:
                axs[3].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class OSCD100(OSCD):
    """Subset of OSCD with 100 pre-cropped image pairs at 256x256 resolution.

    Intended for tutorials and demonstrations, not benchmarking.

    Maintains the same file structure and all 13 Sentinel-2 bands as OSCD,
    but with 100 pre-cropped 256x256 patches. Adds a validation split
    (train/val/test).

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.1109/IGARSS.2018.8518015
    """

    urls = {
        "Onera Satellite Change Detection dataset - Images.zip": "https://hf.co/datasets/hkristen/oscd100/resolve/81edcad799419465bf9ca137281bb72a6f4e4b34/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images.zip",
        "Onera Satellite Change Detection dataset - Train Labels.zip": "https://hf.co/datasets/hkristen/oscd100/resolve/81edcad799419465bf9ca137281bb72a6f4e4b34/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Train%20Labels.zip",
        "Onera Satellite Change Detection dataset - Val Labels.zip": "https://hf.co/datasets/hkristen/oscd100/resolve/81edcad799419465bf9ca137281bb72a6f4e4b34/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Val%20Labels.zip",
        "Onera Satellite Change Detection dataset - Test Labels.zip": "https://hf.co/datasets/hkristen/oscd100/resolve/81edcad799419465bf9ca137281bb72a6f4e4b34/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Test%20Labels.zip",
    }
    sha256s = {
        "Onera Satellite Change Detection dataset - Images.zip": "6c88242b15f3295a1062bea90adffb76af87a4e8b386a0cbb82e5a01663f9480",
        "Onera Satellite Change Detection dataset - Train Labels.zip": "84145c36f55753f182a782c00e62107495e0a44effce90086fad1c5efe5b3d98",
        "Onera Satellite Change Detection dataset - Val Labels.zip": "3a7e8cecacc9be4a12a6d2fc7f7926871039d78c346969c709690869c322f596",
        "Onera Satellite Change Detection dataset - Test Labels.zip": "425571643f1c456caf3cb250f13cbfbbeee247838dfc1e0c50cdb712d8bbf383",
    }
    splits = ("train", "val", "test")
