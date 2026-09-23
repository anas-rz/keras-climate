"""CaFFe dataset (ported from torchgeo.datasets.caffe)."""

import glob
import os
import textwrap

import numpy as np
from keras import ops
from matplotlib.colors import ListedColormap

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive, extract_archive


class CaFFe(NonGeoDataset):
    """CaFFe (CAlving Fronts and where to Find thEm) dataset.

    The `CaFFe <https://doi.pangaea.de/10.1594/PANGAEA.940950>`__ dataset is a
    semantic segmentation dataset of marine-terminating glaciers.

    Dataset features:

    * 13,090 train, 2,241 validation, and 3,761 test images
    * varying spatial resolution of 6-20m
    * paired binary calving front segmentation masks
    * paired multi-class land cover segmentation masks

    Dataset format:

    * images are single-channel pngs with dimension 512x512
    * segmentation masks are single-channel pngs

    Dataset classes:

    0. N/A
    1. rock
    2. glacier
    3. ocean/ice melange

    If you use this dataset in your research, please cite the following paper:

    * https://essd.copernicus.org/articles/14/4287/2022/
    """

    valid_splits = ("train", "val", "test")

    zipfilename = "caffe.zip"

    data_dir = "caffe"

    image_dir = "sar_images"

    mask_dirs = ("fronts", "zones")

    url = "https://huggingface.co/datasets/torchgeo/caffe/resolve/cc96e8418981ce0f03afc9beace6422fdd7142c4/caffe.zip"

    md5 = "9a92fd6f05af74fbc41602595a55df0d"

    px_class_values_zones = {0: "N/A", 64: "rock", 127: "glacier", 254: "ocean/ice melange"}

    zone_class_colors = ("black", "brown", "lightgray", "blue")
    zone_cmap = ListedColormap(zone_class_colors)

    px_class_values_fronts = {0: "no front", 255: "front"}

    def __init__(self, root="data", split="train", transforms=None, download=False, checksum=True):
        """Initialize a new instance of CaFFe dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert split in self.valid_splits, f"split must be one of {self.valid_splits}"

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.fpaths = glob.glob(
            os.path.join(
                self.root,
                self.zipfilename.replace(".zip", ""),
                self.mask_dirs[1],
                self.split,
                "*.png",
            )
        )

        self.ordinal_map_zones = np.zeros(
            max(self.px_class_values_zones.keys()) + 1, dtype="int64"
        )
        for ordinal, px_class in enumerate(self.px_class_values_zones.keys()):
            self.ordinal_map_zones[px_class] = ordinal

        self.ordinal_map_fronts = np.zeros(
            max(self.px_class_values_fronts.keys()) + 1, dtype="int64"
        )
        for ordinal, px_class in enumerate(self.px_class_values_fronts.keys()):
            self.ordinal_map_fronts[px_class] = ordinal

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index):
        zones_filename = os.path.basename(self.fpaths[index])
        img_filename = zones_filename.replace("_zones_", "_")
        front_filename = zones_filename.replace("_zones_", "_front_")

        def read_array(path):
            from PIL import Image

            return np.array(Image.open(path))

        img_path = os.path.join(self.root, self.data_dir, self.image_dir, self.split, img_filename)
        img = read_array(img_path).astype("float32")
        img = img[..., np.newaxis]

        front_mask = read_array(
            os.path.join(self.root, self.data_dir, self.mask_dirs[0], self.split, front_filename)
        ).astype("int64")

        zone_mask = read_array(
            os.path.join(self.root, self.data_dir, self.mask_dirs[1], self.split, zones_filename)
        ).astype("int64")

        zone_mask = self.ordinal_map_zones[zone_mask]
        front_mask = self.ordinal_map_fronts[front_mask]

        sample = {
            "image": ops.convert_to_tensor(img),
            "mask_front": ops.convert_to_tensor(front_mask),
            "mask_zones": ops.convert_to_tensor(zone_mask),
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        exists = []
        if os.path.exists(
            os.path.join(self.root, self.zipfilename.replace(".zip", ""), self.image_dir, self.split)
        ):
            exists.append(True)
        else:
            exists.append(False)

        for mask_dir in self.mask_dirs:
            if os.path.exists(
                os.path.join(self.root, self.zipfilename.replace(".zip", ""), mask_dir, self.split)
            ):
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            return

        if os.path.exists(os.path.join(self.root, self.zipfilename)):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the dataset."""
        download_and_extract_archive(
            self.url, self.root, filename=self.zipfilename, md5=self.md5 if self.checksum else None
        )

    def _extract(self):
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.zipfilename), self.root)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        if "prediction" in sample:
            ncols = 4
        else:
            ncols = 3
        fig, axs = plt.subplots(1, ncols, figsize=(15, 5))

        image = ops.convert_to_numpy(sample["image"])
        mask_front = ops.convert_to_numpy(sample["mask_front"])
        mask_zones = ops.convert_to_numpy(sample["mask_zones"])

        axs[0].imshow(image[..., 0].astype("uint8"), cmap="gray")
        axs[0].axis("off")

        axs[1].imshow(mask_front, cmap="gray")
        axs[1].axis("off")

        unique_classes = np.unique(mask_zones)
        axs[2].imshow(mask_zones, cmap=self.zone_cmap)
        axs[2].axis("off")

        handles = [
            mpatches.Patch(
                color=self.zone_cmap(ordinal),
                label="\n".join(textwrap.wrap(self.px_class_values_zones[px_class], width=10)),
            )
            for ordinal, px_class in enumerate(self.px_class_values_zones.keys())
            if ordinal in unique_classes
        ]
        axs[2].legend(handles=handles, loc="upper right", bbox_to_anchor=(1.4, 1))

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Front Mask")
            axs[2].set_title("Zone Mask")

        if "prediction" in sample:
            axs[3].imshow(ops.convert_to_numpy(sample["prediction"]), cmap="gray")
            axs[3].axis("off")
            if show_titles:
                axs[3].set_title("Prediction")

        if suptitle:
            fig.suptitle(suptitle)

        return fig
