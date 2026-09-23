"""LEVIR-CD and LEVIR-CD+ datasets (ported from torchgeo.datasets.levircd)."""

import abc
import glob
import os

import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive, quantile_normalization


class LEVIRCDBase(NonGeoDataset, abc.ABC):
    """Abstract base class for the LEVIRCD datasets."""

    splits = ()
    directories = ("A", "B", "label")

    def __init__(
        self, root="data", split="train", transforms=None, download=False, checksum=True
    ):
        """Initialize a new LEVIR-CD base dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        self.files = self._load_files(self.root, self.split)

    def __getitem__(self, index):
        """Return an index within the dataset.

        The returned image has shape ``[T, H, W, C]`` where ``T=2`` (before,
        after). The mask has shape ``[H, W]``.
        """
        files = self.files[index]
        image1 = self._load_image(files["image1"])
        image2 = self._load_image(files["image2"])
        mask = self._load_target(files["mask"])
        sample = {"image": ops.stack([image1, image2]), "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def _load_image(self, path):
        """Load a single image."""
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("RGB"))
            return ops.cast(ops.convert_to_tensor(array), "float32")

    def _load_target(self, path):
        """Load the target mask for a single image."""
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("L"))
            array = np.clip(array, 0, 1)
            return ops.cast(ops.convert_to_tensor(array), "int64")

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        ncols = 3

        image1 = quantile_normalization(sample["image"][0])
        image2 = quantile_normalization(sample["image"][1])

        if "prediction" in sample:
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 10))

        axs[0].imshow(ops.convert_to_numpy(image1))
        axs[0].axis("off")
        axs[1].imshow(ops.convert_to_numpy(image2))
        axs[1].axis("off")
        axs[2].imshow(ops.convert_to_numpy(sample["mask"]), cmap="gray", interpolation="none")
        axs[2].axis("off")

        if "prediction" in sample:
            axs[3].imshow(
                ops.convert_to_numpy(sample["prediction"]), cmap="gray", interpolation="none"
            )
            axs[3].axis("off")
            if show_titles:
                axs[3].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image 1")
            axs[1].set_title("Image 2")
            axs[2].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig

    @abc.abstractmethod
    def _load_files(self, root, split):
        """Return the paths of the files in the dataset."""

    @abc.abstractmethod
    def _check_integrity(self):
        """Check the integrity of the dataset structure."""

    @abc.abstractmethod
    def _download(self):
        """Download the dataset and extract it."""


class LEVIRCD(LEVIRCDBase):
    """LEVIR-CD dataset.

    The `LEVIR-CD <https://github.com/justchenhao/STANet>`__ dataset is a
    dataset for building change detection.

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs where no change = 0, change = 255

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.3390/rs12101662
    """

    splits = {
        "train": {
            "url": "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/6a6bb0a5b389403d81c05e33bf08bc0b9e5f13a6/train.zip",
            "filename": "train.zip",
            "md5": "a638e71f480628652dea78d8544307e4",
        },
        "val": {
            "url": "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/6a6bb0a5b389403d81c05e33bf08bc0b9e5f13a6/val.zip",
            "filename": "val.zip",
            "md5": "f7b857978524f9aa8c3bf7f94e3047a4",
        },
        "test": {
            "url": "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/6a6bb0a5b389403d81c05e33bf08bc0b9e5f13a6/test.zip",
            "filename": "test.zip",
            "md5": "07d5dd89e46f5c1359e2eca746989ed9",
        },
    }

    def _load_files(self, root, split):
        """Return the paths of the files in the dataset."""
        images1 = sorted(glob.glob(os.path.join(root, "A", f"{split}*.png")))
        images2 = sorted(glob.glob(os.path.join(root, "B", f"{split}*.png")))
        masks = sorted(glob.glob(os.path.join(root, "label", f"{split}*.png")))

        files = []
        for image1, image2, mask in zip(images1, images2, masks):
            files.append({"image1": image1, "image2": image2, "mask": mask})
        return files

    def _check_integrity(self):
        """Check the integrity of the dataset structure."""
        return all(
            os.path.exists(os.path.join(self.root, directory))
            for directory in self.directories
        )

    def _download(self):
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for split in self.splits:
            download_and_extract_archive(
                self.splits[split]["url"],
                self.root,
                filename=self.splits[split]["filename"],
                md5=self.splits[split]["md5"] if self.checksum else None,
            )


class LEVIRCDPlus(LEVIRCDBase):
    """LEVIR-CD+ dataset.

    The `LEVIR-CD+ <https://github.com/S2Looking/Dataset>`__ dataset extends
    LEVIR-CD to 985 image pairs and is designed to be easier due to its urban
    locations and near-nadir angles.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2107.09244
    """

    url = "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/d4f83dcbb571ee7573079129a5c327d18592a849/LEVIR-CD+.zip"
    md5 = "1adf156f628aa32fb2e8fe6cada16c04"
    filename = "LEVIR-CD+.zip"
    directory = "LEVIR-CD+"
    splits = ("train", "test")

    def _load_files(self, root, split):
        """Return the paths of the files in the dataset."""
        files = []
        images = glob.glob(os.path.join(root, self.directory, split, "A", "*.png"))
        images = sorted(os.path.basename(image) for image in images)
        for image in images:
            image1 = os.path.join(root, self.directory, split, "A", image)
            image2 = os.path.join(root, self.directory, split, "B", image)
            mask = os.path.join(root, self.directory, split, "label", image)
            files.append({"image1": image1, "image2": image2, "mask": mask})
        return files

    def _check_integrity(self):
        """Check the integrity of the dataset structure."""
        for filename in self.splits:
            filepath = os.path.join(self.root, self.directory, filename)
            if not os.path.exists(filepath):
                return False
        return True

    def _download(self):
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.md5 if self.checksum else None
        )
