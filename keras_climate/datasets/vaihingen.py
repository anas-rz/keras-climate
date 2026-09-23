"""Vaihingen dataset (ported from torchgeo.datasets.vaihingen)."""

import os

import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, draw_semantic_segmentation_masks, extract_archive, rgb_to_mask


class Vaihingen2D(NonGeoDataset):
    """Vaihingen 2D Semantic Segmentation dataset.

    The `Vaihingen <https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx>`__
    dataset is a dataset for urban semantic segmentation used in the 2D
    Semantic Labeling Contest - Vaihingen. This dataset uses the
    "ISPRS_semantic_labeling_Vaihingen.zip" and
    "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip" files to
    create the train/test sets used in the challenge. The dataset can be
    downloaded from `here
    <https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx>`__.
    Note, the server contains additional data for 3D Semantic Labeling which
    are currently not supported.

    Dataset format:

    * images are 3-channel RGB geotiffs
    * masks are 3-channel geotiffs with unique RGB values representing the
      class

    Dataset classes:

    0. Clutter/background
    1. Impervious surfaces
    2. Building
    3. Low Vegetation
    4. Tree
    5. Car

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.5194/isprsannals-I-3-293-2012
    """

    filenames = (
        "ISPRS_semantic_labeling_Vaihingen.zip",
        "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip",
    )
    md5s = ("462b8dca7b6fa9eaf729840f0cdfc7f3", "4802dd6326e2727a352fb735be450277")
    image_root = "top"
    splits = {
        "train": [
            "top_mosaic_09cm_area1.tif",
            "top_mosaic_09cm_area11.tif",
            "top_mosaic_09cm_area13.tif",
            "top_mosaic_09cm_area15.tif",
            "top_mosaic_09cm_area17.tif",
            "top_mosaic_09cm_area21.tif",
            "top_mosaic_09cm_area23.tif",
            "top_mosaic_09cm_area26.tif",
            "top_mosaic_09cm_area28.tif",
            "top_mosaic_09cm_area3.tif",
            "top_mosaic_09cm_area30.tif",
            "top_mosaic_09cm_area32.tif",
            "top_mosaic_09cm_area34.tif",
            "top_mosaic_09cm_area37.tif",
            "top_mosaic_09cm_area5.tif",
            "top_mosaic_09cm_area7.tif",
        ],
        "test": [
            "top_mosaic_09cm_area6.tif",
            "top_mosaic_09cm_area24.tif",
            "top_mosaic_09cm_area35.tif",
            "top_mosaic_09cm_area16.tif",
            "top_mosaic_09cm_area14.tif",
            "top_mosaic_09cm_area22.tif",
            "top_mosaic_09cm_area10.tif",
            "top_mosaic_09cm_area4.tif",
            "top_mosaic_09cm_area2.tif",
            "top_mosaic_09cm_area20.tif",
            "top_mosaic_09cm_area8.tif",
            "top_mosaic_09cm_area31.tif",
            "top_mosaic_09cm_area33.tif",
            "top_mosaic_09cm_area27.tif",
            "top_mosaic_09cm_area38.tif",
            "top_mosaic_09cm_area12.tif",
            "top_mosaic_09cm_area29.tif",
        ],
    }
    classes = (
        "Clutter/background",
        "Impervious surfaces",
        "Building",
        "Low Vegetation",
        "Tree",
        "Car",
    )
    colormap = (
        (255, 0, 0),
        (255, 255, 255),
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
    )

    def __init__(self, root="data", split="train", transforms=None, checksum=True):
        """Initialize a new Vaihingen2D dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: If *split* is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert split in self.splits
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.files = []
        for name in self.splits[split]:
            image = os.path.join(root, self.image_root, name)
            mask = os.path.join(root, name)
            if os.path.exists(image) and os.path.exists(mask):
                self.files.append({"image": image, "mask": mask})

    def __getitem__(self, index):
        """Return an index within the dataset."""
        image = self._load_image(index)
        mask = self._load_target(index)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def _load_image(self, index):
        """Load a single image."""
        path = self.files[index]["image"]
        with Image.open(path) as img:
            array = np.array(img.convert("RGB")).astype("float32")
        return ops.convert_to_tensor(array)

    def _load_target(self, index):
        """Load the target mask for a single image."""
        path = self.files[index]["mask"]
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
            array = rgb_to_mask(array, self.colormap)
        return ops.convert_to_tensor(array.astype("int64"))

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.image_root)):
            return

        # Check if .zip files already exists (if so extract)
        exists = []
        for filename, md5 in zip(self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, md5):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        raise DatasetNotFoundError(self)

    def plot(self, sample, show_titles=True, suptitle=None, alpha=0.5):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        ncols = 1
        image1 = draw_semantic_segmentation_masks(
            sample["image"][:, :, :3], sample["mask"], alpha=alpha, colors=list(self.colormap)
        )
        if "prediction" in sample:
            ncols += 1
            image2 = draw_semantic_segmentation_masks(
                sample["image"][:, :, :3],
                sample["prediction"],
                alpha=alpha,
                colors=list(self.colormap),
            )

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if ncols > 1:
            (ax0, ax1) = axs
        else:
            ax0 = axs

        ax0.imshow(image1)
        ax0.axis("off")
        if ncols > 1:
            ax1.imshow(image2)
            ax1.axis("off")

        if show_titles:
            ax0.set_title("Ground Truth")
            if ncols > 1:
                ax1.set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
