"""Potsdam dataset (ported from torchgeo.datasets.potsdam)."""

import os

import numpy as np
import rasterio
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, draw_semantic_segmentation_masks, extract_archive, rgb_to_mask


class Potsdam2D(NonGeoDataset):
    """Potsdam 2D Semantic Segmentation dataset.

    The `Potsdam <https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx>`__
    dataset is a dataset for urban semantic segmentation used in the 2D
    Semantic Labeling Contest - Potsdam. This dataset uses the
    "4_Ortho_RGBIR.zip" and "5_Labels_all.zip" files to create the
    train/test sets used in the challenge. The dataset can be requested at
    the challenge homepage.

    Dataset format:

    * images are 4-channel geotiffs
    * masks are 3-channel geotiffs with unique RGB values representing the
      class

    Dataset classes:

    0. Clutter/background
    1. Impervious surfaces
    2. Building
    3. Low Vegetation
    4. Tree
    5. Car

    If you use this dataset in your research, please cite:

    * https://doi.org/10.5194/isprsannals-I-3-293-2012
    """

    filenames = ("4_Ortho_RGBIR.zip", "5_Labels_all.zip")
    md5s = ("c4a8f7d8c7196dd4eba4addd0aae10c1", "cf7403c1a97c0d279414db")
    image_root = "4_Ortho_RGBIR"
    splits = {
        "train": [
            "top_potsdam_2_10",
            "top_potsdam_2_11",
            "top_potsdam_2_12",
            "top_potsdam_3_10",
            "top_potsdam_3_11",
            "top_potsdam_3_12",
            "top_potsdam_4_10",
            "top_potsdam_4_11",
            "top_potsdam_4_12",
            "top_potsdam_5_10",
            "top_potsdam_5_11",
            "top_potsdam_5_12",
            "top_potsdam_6_10",
            "top_potsdam_6_11",
            "top_potsdam_6_12",
            "top_potsdam_6_7",
            "top_potsdam_6_8",
            "top_potsdam_6_9",
            "top_potsdam_7_10",
            "top_potsdam_7_11",
            "top_potsdam_7_12",
            "top_potsdam_7_7",
            "top_potsdam_7_8",
            "top_potsdam_7_9",
        ],
        "test": [
            "top_potsdam_5_15",
            "top_potsdam_6_15",
            "top_potsdam_6_13",
            "top_potsdam_3_13",
            "top_potsdam_4_14",
            "top_potsdam_6_14",
            "top_potsdam_5_14",
            "top_potsdam_2_13",
            "top_potsdam_4_15",
            "top_potsdam_2_14",
            "top_potsdam_5_13",
            "top_potsdam_4_13",
            "top_potsdam_3_14",
            "top_potsdam_7_13",
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
        """Initialize a new Potsdam dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: If *split* is invalid.
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in self.splits
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.files = []
        for name in self.splits[split]:
            image = os.path.join(root, self.image_root, name) + "_RGBIR.tif"
            mask = os.path.join(root, name) + "_label.tif"
            if os.path.exists(image) and os.path.exists(mask):
                self.files.append({"image": image, "mask": mask})

    def __getitem__(self, index):
        image = self._load_image(index)
        mask = self._load_target(index)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _load_image(self, index):
        path = self.files[index]["image"]
        with rasterio.open(path) as f:
            array = f.read()
            array = np.transpose(array, (1, 2, 0)).astype("float32")
            return ops.convert_to_tensor(array)

    def _load_target(self, index):
        path = self.files[index]["mask"]
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
            array = rgb_to_mask(array, self.colormap)
            array = array.astype("int64")
        return ops.convert_to_tensor(array)

    def _verify(self):
        """Verify the integrity of the dataset."""
        if os.path.exists(os.path.join(self.root, self.image_root)):
            return

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
        image = ops.convert_to_numpy(sample["image"])[..., :3].astype("uint8")
        mask = ops.convert_to_numpy(sample["mask"])
        image1 = draw_semantic_segmentation_masks(
            ops.convert_to_tensor(image), mask, alpha=alpha, colors=list(self.colormap)
        )
        if "prediction" in sample:
            ncols += 1
            prediction = ops.convert_to_numpy(sample["prediction"])
            image2 = draw_semantic_segmentation_masks(
                ops.convert_to_tensor(image),
                prediction,
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
