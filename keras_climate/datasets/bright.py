"""BRIGHT dataset (ported from torchgeo.datasets.bright)."""

import os
import textwrap

import numpy as np
import rasterio
from keras import ops
from matplotlib import colors

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_url, extract_archive


class BRIGHTDFC2025(NonGeoDataset):
    """BRIGHT DFC2025 dataset.

    The `BRIGHT <https://github.com/ChenHongruixuan/BRIGHT>`__ dataset
    consists of bi-temporal high-resolution multimodal images for building
    damage assessment. The dataset is part of the 2025 IEEE GRSS Data Fusion
    Contest. The pre-disaster images are optical images and the
    post-disaster images are SAR images, and targets were manually
    annotated. The dataset is split into train, val, and test splits, but
    the test split does not contain targets in this version.

    More information can be found at the `Challenge website
    <https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/?tab=data-fusion-contest>`__.

    Dataset Features:

    * Pre-disaster optical images from MAXAR, NAIP, NOAA Digital Coast
      Raster Datasets, and the National Plan for Aerial Orthophotography
      Spain
    * Post-disaster SAR images from Capella Space and Umbra
    * high image resolution of 0.3-1m

    Dataset Format:

    * Images are in GeoTIFF format with pixel dimensions of 1024x1024
    * Pre-disaster are three channel images
    * Post-disaster SAR images are single channel but repeated to have 3
      channels

    If you use this dataset in your research, please cite the following
    paper:

    * https://arxiv.org/abs/2501.06019
    """

    classes = ("background", "intact", "damaged", "destroyed")

    colormap = (
        "white",  # background
        "green",  # intact
        "burlywood",  # damaged
        "red",  # destroyed
    )

    sha256 = "aed4aca4d582bc829ba0921b55d488eec6a792c2d1092aac54eb746ef8e6f1cc"

    url = "https://hf.co/datasets/isaaccorley/bright/resolve/d19972f5e682ad684dcde35529a6afad4c719f1b/dfc25_track2_trainval_with_split.zip"

    data_dir = "dfc25_track2_trainval"

    valid_split = ("train", "val", "test")

    # train_setlevels.txt are the training samples
    # holdout_setlevels.txt are the validation samples
    # val_setlevels.txt are the test samples
    split_files = {
        "train": "train_setlevel.txt",
        "val": "holdout_setlevel.txt",
        "test": "val_setlevel.txt",
    }

    px_class_values = {0: "background", 1: "intact", 2: "damaged", 3: "destroyed"}

    def __init__(
        self, root="data", split="train", transforms=None, download=False, checksum=True
    ):
        """Initialize a new BRIGHT DFC2025 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
            AssertionError: If *split* is not one of 'train', 'val', or
                'test.
        """
        assert split in self.valid_split, f"Split must be one of {self.valid_split}"
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.sample_paths = self._get_paths()

    def __getitem__(self, index):
        """Return an index within the dataset.

        Returns:
            data and target at that index, pre and post image are stacked
            along a leading time axis (2, H, W, C)
        """
        idx_paths = self.sample_paths[index]

        image_pre = ops.cast(self._load_image(idx_paths["image_pre"]), "float32")
        image_post = ops.cast(self._load_image(idx_paths["image_post"]), "float32")
        # https://github.com/ChenHongruixuan/BRIGHT/blob/11b1ffafa4d30d2df2081189b56864b0de4e3ed7/dfc25_benchmark/dataset/make_data_loader.py#L101
        # post image is repeated to also have 3 channels
        image_post = ops.tile(image_post, (1, 1, 3))

        sample = {"image": ops.stack([image_pre, image_post])}

        if "target" in idx_paths and self.split != "test":
            target = ops.cast(self._load_image(idx_paths["target"]), "int64")
            # single-band mask: squeeze the channel dimension
            sample["mask"] = ops.squeeze(target, axis=-1)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _get_paths(self):
        """Get paths to the dataset files based on specified splits."""
        split_file = self.split_files[self.split]

        file_path = os.path.join(self.root, self.data_dir, split_file)
        with open(file_path) as f:
            sample_ids = f.readlines()

        if self.split in ("train", "val"):
            dir_split_name = "train"
        else:
            dir_split_name = "val"

        sample_paths = [
            {
                "image_pre": os.path.join(
                    self.root,
                    self.data_dir,
                    dir_split_name,
                    "pre-event",
                    f"{sample_id.strip()}_pre_disaster.tif",
                ),
                "image_post": os.path.join(
                    self.root,
                    self.data_dir,
                    dir_split_name,
                    "post-event",
                    f"{sample_id.strip()}_post_disaster.tif",
                ),
            }
            for sample_id in sample_ids
        ]
        if self.split != "test":
            for sample, sample_id in zip(sample_paths, sample_ids):
                sample["target"] = os.path.join(
                    self.root,
                    self.data_dir,
                    dir_split_name,
                    "target",
                    f"{sample_id.strip()}_building_damage.tif",
                )

        return sample_paths

    def _verify(self):
        """Verify the integrity of the dataset."""
        if all(
            os.path.exists(os.path.join(self.root, self.data_dir, split_file))
            for split_file in self.split_files.values()
        ):
            sample_paths = self._get_paths()
            exists = []
            for sample in sample_paths:
                exists.append(
                    all(os.path.exists(path) for name, path in sample.items())
                )
            if all(exists):
                return

        exists = []
        zip_file_path = os.path.join(self.root, self.data_dir + ".zip")
        if os.path.exists(zip_file_path):
            if self.checksum and not check_integrity(
                zip_file_path, sha256=self.sha256
            ):
                raise RuntimeError("Dataset found, but corrupted.")
            exists.append(True)
            extract_archive(zip_file_path, self.root)
        else:
            exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        extract_archive(zip_file_path, self.root)

    def _download(self):
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            self.data_dir + ".zip",
            sha256=self.sha256 if self.checksum else None,
        )

    def __len__(self):
        return len(self.sample_paths)

    def _load_image(self, path):
        """Load a file from disk, returned channels-last (H, W, C)."""
        with rasterio.open(path) as src:
            img = np.transpose(src.read(), (1, 2, 0))
        return ops.convert_to_tensor(img)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        ncols = 2
        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample
        if showing_mask:
            ncols += 1
        if showing_prediction:
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5))

        image = ops.convert_to_numpy(sample["image"])

        axs[0].imshow(image[0] / 255.0)
        axs[0].axis("off")

        axs[1].imshow(image[1] / 255.0)
        axs[1].axis("off")

        cmap = colors.ListedColormap(self.colormap)

        kwargs = {"cmap": cmap, "vmin": 0, "vmax": 3, "interpolation": "none"}
        if showing_mask:
            mask = ops.convert_to_numpy(sample["mask"])
            axs[2].imshow(mask, **kwargs)
            axs[2].axis("off")
            unique_classes = np.unique(mask)
            handles = [
                mpatches.Patch(
                    color=cmap(ordinal),
                    label="\n".join(
                        textwrap.wrap(self.px_class_values[px_class], width=10)
                    ),
                )
                for ordinal, px_class in enumerate(self.px_class_values.keys())
                if ordinal in unique_classes
            ]
            axs[2].legend(handles=handles, loc="upper right", bbox_to_anchor=(1.4, 1))
            if showing_prediction:
                axs[3].imshow(ops.convert_to_numpy(sample["prediction"]), **kwargs)
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(ops.convert_to_numpy(sample["prediction"]), **kwargs)
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Pre-disaster image")
            axs[1].set_title("Post-disaster image")
            if showing_mask:
                axs[2].set_title("Ground truth")
            if showing_prediction:
                axs[-1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
