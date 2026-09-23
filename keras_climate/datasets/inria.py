"""Inria Aerial Image Labeling Dataset (ported from torchgeo.datasets.inria)."""

import glob
import os
import re

import numpy as np
import rasterio as rio
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive, quantile_normalization


class InriaAerialImageLabeling(NonGeoDataset):
    r"""Inria Aerial Image Labeling Dataset.

    The `Inria Aerial Image Labeling <https://project.inria.fr/aerialimagelabeling/>`__
    dataset is a building detection dataset over dissimilar settlements
    ranging from densely populated areas to alpine towns. Refer to the
    dataset homepage to download the dataset.

    Dataset features:

    * Coverage of 810 km2 (405 km2 for training and 405 km2 for testing)
    * Aerial orthorectified color imagery with a spatial resolution of 0.3 m
    * Number of images: 360 (train: 180, test: 180)
    * Train cities: Austin, Chicago, Kitsap, West Tyrol, Vienna
    * Test cities: Bellingham, Bloomington, Innsbruck, San Francisco, East Tyrol

    Dataset format:

    * Imagery - RGB aerial GeoTIFFs of shape 5000 x 5000
    * Labels - RGB aerial GeoTIFFs of shape 5000 x 5000

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.1109/IGARSS.2017.8127684
    """

    directory = "AerialImageDataset"
    filename = "NEW2-AerialImageDataset.zip"
    md5 = "4b1acfe84ae9961edc1a6049f940380f"

    def __init__(self, root="data", split="train", transforms=None, checksum=True):
        """Initialize a new InriaAerialImageLabeling Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version.
            checksum: if True, check the MD5 of the downloaded files

        Raises:
            AssertionError: if ``split`` is invalid
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        assert split in {"train", "val", "test"}
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()
        self.files = self._load_files(root)

    def _load_files(self, root):
        """Return the paths of the files in the dataset."""
        files = []
        split = "train" if self.split in ["train", "val"] else "test"
        root_dir = os.path.join(root, self.directory, split)
        pattern = re.compile(r"([A-Za-z]+)(\d+)")

        images = glob.glob(os.path.join(root_dir, "images", "*.tif"))
        images = sorted(images)

        if split == "train":
            labels = glob.glob(os.path.join(root_dir, "gt", "*.tif"))
            labels = sorted(labels)

            for img, lbl in zip(images, labels):
                fname = os.path.basename(img)
                if match := pattern.search(fname):
                    idx = int(match.group(2))
                    # For validation, use the first 5 images of every location
                    if (self.split == "train" and idx > 5) or (
                        self.split == "val" and idx < 6
                    ):
                        files.append({"image": img, "label": lbl})
        else:
            for img in images:
                files.append({"image": img})

        return files

    def _load_image(self, path):
        """Load a single image."""
        with rio.open(path) as img:
            array = img.read().astype(np.int32)
            array = np.transpose(array, (1, 2, 0))
            tensor = ops.convert_to_tensor(array.astype("float32"))
            return tensor

    def _load_target(self, path):
        """Loads the target mask."""
        with rio.open(path) as img:
            array = img.read().astype(np.int32)
            array = np.clip(array, 0, 1)
            mask = ops.convert_to_tensor(array[0].astype("int64"))
            return mask

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.files)

    def __getitem__(self, index):
        """Return an index within the dataset."""
        files = self.files[index]
        img = self._load_image(files["image"])
        sample = {"image": img}
        if files.get("label"):
            mask = self._load_target(files["label"])
            sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Checks the integrity of the dataset structure."""
        if os.path.isdir(os.path.join(self.root, self.directory)):
            return

        archive_path = os.path.join(self.root, self.filename)
        md5_hash = self.md5 if self.checksum else None
        if not os.path.isfile(archive_path):
            raise DatasetNotFoundError(self)
        if not check_integrity(archive_path, md5_hash):
            raise RuntimeError("Dataset corrupted")
        print("Extracting...")
        extract_archive(archive_path)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])[..., :3]
        image = quantile_normalization(image)
        image = ops.convert_to_numpy(image)

        ncols = 1
        show_mask = "mask" in sample
        show_predictions = "prediction" in sample

        if show_mask:
            mask = ops.convert_to_numpy(sample["mask"])
            ncols += 1

        if show_predictions:
            prediction = ops.convert_to_numpy(sample["prediction"])
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis("off")
        if show_titles:
            axs[0].set_title("Image")

        if show_mask:
            axs[1].imshow(mask, interpolation="none")
            axs[1].axis("off")
            if show_titles:
                axs[1].set_title("Label")

        if show_predictions:
            axs[2].imshow(prediction, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
