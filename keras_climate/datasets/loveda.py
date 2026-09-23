"""LoveDA dataset (ported from torchgeo.datasets.loveda)."""

import glob
import os

import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive


class LoveDA(NonGeoDataset):
    """LoveDA dataset.

    The `LoveDA <https://github.com/Junjue-Wang/LoveDA>`__ dataset is a
    semantic segmentation dataset.

    Dataset features:

    * 2713 urban scene and 3274 rural scene HSR images, spatial resolution
      of 0.3m
    * dataset comes with predefined train, validation, and test set
    * dataset differentiates between 'rural' and 'urban' images

    Dataset classes:

    1. background
    2. building
    3. road
    4. water
    5. barren
    6. forest
    7. agriculture

    No-data regions assigned with 0 and should be ignored.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2110.08733
    """

    scenes = ("urban", "rural")
    splits = ("train", "val", "test")

    info_dict = {
        "train": {
            "url": "https://zenodo.org/records/5706578/files/Train.zip?download=1",
            "filename": "Train.zip",
            "md5": "de2b196043ed9b4af1690b3f9a7d558f",
        },
        "val": {
            "url": "https://zenodo.org/records/5706578/files/Val.zip?download=1",
            "filename": "Val.zip",
            "md5": "84cae2577468ff0b5386758bb386d31d",
        },
        "test": {
            "url": "https://zenodo.org/records/5706578/files/Test.zip?download=1",
            "filename": "Test.zip",
            "md5": "a489be0090465e01fb067795d24e6b47",
        },
    }

    classes = (
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
        "no-data",
    )

    def __init__(
        self,
        root="data",
        split="train",
        scene=["urban", "rural"],
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new LoveDA dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            scene: specify whether to load only 'urban', only 'rural' or both
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: if ``split`` or ``scene`` arguments are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
        assert set(scene).intersection(set(self.scenes)), (
            "The possible scenes are 'rural' and/or 'urban'"
        )
        assert len(scene) <= 2, "There are no other scenes than 'rural' or 'urban'"

        self.root = root
        self.split = split
        self.scene = scene
        self.transforms = transforms
        self.checksum = checksum

        self.url = self.info_dict[self.split]["url"]
        self.filename = self.info_dict[self.split]["filename"]
        self.md5 = self.info_dict[self.split]["md5"]

        self.directory = os.path.join(self.root, split.capitalize())
        self.scene_paths = [
            os.path.join(self.directory, s.capitalize()) for s in self.scene
        ]

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        self.files = self._load_files(self.scene_paths, self.split)

    def __getitem__(self, index):
        """Return an index within the dataset.

        Returns:
            image and mask at that index with image of shape ``[H, W, 3]``
            and mask of shape ``[H, W]``.
        """
        files = self.files[index]
        image = self._load_image(files["image"])

        if self.split != "test":
            mask = self._load_target(files["mask"])
            sample = {"image": image, "mask": mask}
        else:
            sample = {"image": image}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of datapoints in the dataset."""
        return len(self.files)

    def _load_files(self, scene_paths, split):
        """Return the paths of the files in the dataset."""
        images = []

        for s in scene_paths:
            images.extend(glob.glob(os.path.join(s, "images_png", "*.png")))

        images = sorted(images)

        if self.split != "test":
            masks = [image.replace("images_png", "masks_png") for image in images]
            files = [
                {"image": image, "mask": mask} for image, mask in zip(images, masks)
            ]
        else:
            files = [{"image": image} for image in images]

        return files

    def _load_image(self, path):
        """Load a single image."""
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("RGB"))
            return ops.cast(ops.convert_to_tensor(array), "float32")

    def _load_target(self, path):
        """Load a single mask corresponding to image."""
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("L"))
            return ops.cast(ops.convert_to_tensor(array), "int64")

    def _check_integrity(self):
        """Check the integrity of the dataset structure."""
        for s in self.scene_paths:
            if not os.path.exists(s):
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

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        if self.split != "test":
            image, mask = sample["image"], sample["mask"]
            ncols = 2
        else:
            image = sample["image"]
            ncols = 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 10, 10))

        image_np = ops.convert_to_numpy(image).astype("uint8")

        if self.split != "test":
            axs[0].imshow(image_np)
            axs[0].axis("off")
            axs[1].imshow(ops.convert_to_numpy(mask))
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Image")
                axs[1].set_title("Mask")
        else:
            axs.imshow(image_np)
            axs.axis("off")
            if show_titles:
                axs.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
