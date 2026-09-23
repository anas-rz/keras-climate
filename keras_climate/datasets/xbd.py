"""xBD dataset (ported from torchgeo.datasets.xbd)."""

import glob
import os

import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, draw_semantic_segmentation_masks, extract_archive


def _colors_to_rgb(colors):
    """Convert named matplotlib colors to 0-255 RGB int tuples."""
    from matplotlib.colors import to_rgb

    return [tuple(int(round(c * 255)) for c in to_rgb(color)) for color in colors]


class xBD(NonGeoDataset):
    """xBD dataset.

    The `xBD <https://xview2.org/dataset>`__ dataset is a dataset for
    building disaster change detection. This dataset object uses the
    "Challenge training set (~7.8 GB)" and "Challenge test set (~2.6 GB)"
    data from the xView2 website as the train and test splits. Note, the
    xView2 website contains other data under the xView2 umbrella that are
    _not_ included here. E.g. the "Tier3 training data", the "Challenge
    holdout set", and the "full data".

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs where the pixel values represent the
      class

    Dataset classes:

    0. background
    1. no damage
    2. minor damage
    3. major damage
    4. destroyed

    If you use this dataset in your research, please cite:

    * https://arxiv.org/abs/1911.09296
    """

    metadata = {
        "train": {
            "filename": "train_images_labels_targets.tar.gz",
            "sha256": "a5941b7a3e523eafc4aeaa740a1c83f1af6a18c894e7e8c62dd830a76921ecd4",
            "directory": "train",
        },
        "test": {
            "filename": "test_images_labels_targets.tar.gz",
            "sha256": "0fcdbfe3ee7d0842729dd2230217e74b2f12be35546ff666df4dae5388e2541c",
            "directory": "test",
        },
    }
    classes = ("background", "no-damage", "minor-damage", "major-damage", "destroyed")
    colormap = ("green", "blue", "orange", "red")

    def __init__(self, root="data", split="train", transforms=None, checksum=True):
        """Initialize a new xBD dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: If *split* is invalid.
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in self.metadata
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files(root, split)

    def __getitem__(self, index):
        """Return an index within the dataset.

        Returns a single T x H x W x C image and a change detection mask.
        """
        sample = self._load_sample(self.files[index])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _load_files(self, root, split):
        """Return the paths of the files in the dataset."""
        files = []
        directory = self.metadata[split]["directory"]
        image_root = os.path.join(root, directory, "images")
        mask_root = os.path.join(root, directory, "targets")
        images = glob.glob(os.path.join(image_root, "*.png"))
        basenames = [os.path.basename(f) for f in images]
        basenames = ["_".join(f.split("_")[:-2]) for f in basenames]
        for name in sorted(set(basenames)):
            image1 = os.path.join(image_root, f"{name}_pre_disaster.png")
            image2 = os.path.join(image_root, f"{name}_post_disaster.png")
            mask1 = os.path.join(mask_root, f"{name}_pre_disaster_target.png")
            mask2 = os.path.join(mask_root, f"{name}_post_disaster_target.png")
            files.append({"image1": image1, "image2": image2, "mask1": mask1, "mask2": mask2})
        return files

    def _load_sample(self, files):
        """Load a sample from a file record."""
        image1 = self._load_image(files["image1"])
        image2 = self._load_image(files["image2"])
        mask1 = self._load_target(files["mask1"])
        mask2 = self._load_target(files["mask2"])

        image = ops.stack([image1, image2], axis=0)
        # Dataset consists of semantic segmentation masks before and after
        # the event. Convert to change detection by subtracting damage
        # before from damage after. Clamp to avoid potential negative
        # numbers.
        mask = ops.clip(ops.subtract(mask2, mask1), 0, 4)
        return {"image": image, "mask": mask}

    def _load_image(self, path):
        """Load a single image, returned channels-last (H, W, C)."""
        with Image.open(path) as img:
            array = np.array(img.convert("RGB")).astype("float32")
            return ops.convert_to_tensor(array)

    def _load_target(self, path):
        """Load the target mask for a single image."""
        with Image.open(path) as img:
            array = np.array(img.convert("L")).astype("int64")
            return ops.convert_to_tensor(array)

    def _verify(self):
        """Verify the integrity of the dataset."""
        exists = []
        for split_info in self.metadata.values():
            for directory in ["images", "targets"]:
                exists.append(os.path.exists(os.path.join(self.root, split_info["directory"], directory)))

        if all(exists):
            return

        # Check if .tar.gz files already exists (if so then extract)
        exists = []
        for split_info in self.metadata.values():
            filepath = os.path.join(self.root, split_info["filename"])
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, sha256=split_info["sha256"]):
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

        ncols = 2
        colors = _colors_to_rgb(self.colormap)
        images = ops.convert_to_numpy(sample["image"])
        image1 = draw_semantic_segmentation_masks(images[0], sample["mask"], alpha=alpha, colors=colors)
        image2 = draw_semantic_segmentation_masks(images[1], sample["mask"], alpha=alpha, colors=colors)
        if "prediction" in sample:
            ncols += 1
            image3 = draw_semantic_segmentation_masks(
                images[1], sample["prediction"], alpha=alpha, colors=colors
            )

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs[0].imshow(image1)
        axs[0].axis("off")
        axs[1].imshow(image2)
        axs[1].axis("off")
        if ncols > 2:
            axs[2].imshow(image3)
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Pre disaster")
            axs[1].set_title("Post disaster")
            if ncols > 2:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class XView2(xBD):
    """Deprecated alias for the xBD dataset.

    .. deprecated:: Use :class:`xBD` instead.
    """


class xBDDistShift(xBD):
    """xBD dataset with a custom, disaster-based train/test split.

    Uses disasters as the shift axis and converts damage masks to binary
    building masks.

    If you use this dataset in your research, please cite:

    * https://arxiv.org/abs/2412.13394
    """

    classes = ("background", "building")
    colormap = ("blue",)
    valid_disasters = (
        "hurricane-harvey",
        "socal-fire",
        "hurricane-matthew",
        "mexico-earthquake",
        "guatemala-volcano",
        "santa-rosa-wildfire",
        "palu-tsunami",
        "hurricane-florence",
        "hurricane-michael",
        "midwest-flooding",
    )

    def __init__(
        self,
        root="data",
        split="train",
        id_disaster="hurricane-matthew",
        id_pre_post="post",
        ood_disaster="mexico-earthquake",
        ood_pre_post="post",
        transforms=None,
        checksum=True,
    ):
        """Initialize a new xBDDistShift dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            id_disaster: disaster used as the in-distribution training set
            id_pre_post: imagery to use for the in-distribution disaster
            ood_disaster: disaster used as the out-of-distribution test set
            ood_pre_post: imagery to use for the out-of-distribution disaster
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: If *split* or the disaster shift configuration is
                invalid.
            DatasetNotFoundError: If dataset is not found.
        """
        assert {id_disaster, ood_disaster} <= set(self.valid_disasters)
        assert id_disaster != ood_disaster
        assert {id_pre_post, ood_pre_post} <= {"pre", "post", "both"}
        self.id_disaster = id_disaster
        self.id_pre_post = id_pre_post
        self.ood_disaster = ood_disaster
        self.ood_pre_post = ood_pre_post
        super().__init__(root, split, transforms, checksum)

    def _load_files(self, root, split):
        """Return files matching the disaster selected for a split."""
        disaster = self.id_disaster if split == "train" else self.ood_disaster
        pre_post = self.id_pre_post if split == "train" else self.ood_pre_post
        files = []
        for split_info in self.metadata.values():
            directory = split_info["directory"]
            image_root = os.path.join(root, directory, "images")
            mask_root = os.path.join(root, directory, "targets")
            for image in sorted(glob.glob(os.path.join(image_root, "*.png"))):
                basename = os.path.basename(image)
                image_disaster = basename.split("_")[0]
                image_pre_post = "pre" if "pre_disaster" in basename else "post"
                if image_disaster != disaster or pre_post not in ("both", image_pre_post):
                    continue

                mask = os.path.join(mask_root, basename.replace(".png", "_target.png"))
                files.append({"image": image, "mask": mask})

        return files

    def _load_sample(self, files):
        """Load a binary building segmentation sample."""
        image = self._load_image(files["image"])
        mask = self._load_target(files["mask"])
        mask = ops.cast(ops.logical_or(ops.equal(mask, 1), ops.equal(mask, 2)), "int64")
        return {"image": image, "mask": mask}

    def plot(self, sample, show_titles=True, suptitle=None, alpha=0.5):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        ncols = 1
        colors = _colors_to_rgb(self.colormap)
        image = draw_semantic_segmentation_masks(sample["image"], sample["mask"], alpha=alpha, colors=colors)
        if "prediction" in sample:
            ncols += 1
            prediction = draw_semantic_segmentation_masks(
                sample["image"], sample["prediction"], alpha=alpha, colors=colors
            )

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10), squeeze=False)
        axs[0, 0].imshow(image)
        axs[0, 0].axis("off")
        if ncols > 1:
            axs[0, 1].imshow(prediction)
            axs[0, 1].axis("off")

        if show_titles:
            axs[0, 0].set_title("Image")
            if ncols > 1:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
