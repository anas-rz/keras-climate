"""2022 IEEE GRSS Data Fusion Contest (DFC2022) dataset (ported from torchgeo.datasets.dfc2022)."""

import glob
import os

import numpy as np
import rasterio
from keras import ops
from rasterio.enums import Resampling

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive, quantile_normalization


class DFC2022(NonGeoDataset):
    """DFC2022 dataset.

    The `DFC2022 <https://www.grss-ieee.org/community/technical-committees/2022-ieee-grss-data-fusion-contest/>`__
    dataset is used as a benchmark dataset for the 2022 IEEE GRSS Data
    Fusion Contest and extends the MiniFrance dataset for semi-supervised
    semantic segmentation. The dataset consists of a train set containing
    labeled and unlabeled imagery and an unlabeled validation set. The
    dataset can be downloaded from the `IEEEDataPort DFC2022 website
    <https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022/>`_.

    Dataset features:

    * RGB aerial images at 0.5 m per pixel spatial resolution (~2,000x2,0000 px)
    * DEMs at 1 m per pixel spatial resolution (~1,000x1,0000 px)
    * Masks at 0.5 m per pixel spatial resolution (~2,000x2,0000 px)
    * 16 land use/land cover categories

    Dataset format:

    * images are three-channel geotiffs
    * DEMS are single-channel geotiffs
    * masks are single-channel geotiffs with the pixel values represent the
      class

    Dataset classes:

    0. No information
    1. Urban fabric
    2. Industrial, commercial, public, military, private and transport units
    3. Mine, dump and construction sites
    4. Artificial non-agricultural vegetated areas
    5. Arable land (annual crops)
    6. Permanent crops
    7. Pastures
    8. Complex and mixed cultivation patterns
    9. Orchards at the fringe of urban classes
    10. Forests
    11. Herbaceous vegetation associations
    12. Open spaces with little or no vegetation
    13. Wetlands
    14. Water
    15. Clouds and Shadows

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1007/s10994-020-05943-y
    """

    classes = (
        "No information",
        "Urban fabric",
        "Industrial, commercial, public, military, private and transport units",
        "Mine, dump and construction sites",
        "Artificial non-agricultural vegetated areas",
        "Arable land (annual crops)",
        "Permanent crops",
        "Pastures",
        "Complex and mixed cultivation patterns",
        "Orchards at the fringe of urban classes",
        "Forests",
        "Herbaceous vegetation associations",
        "Open spaces with little or no vegetation",
        "Wetlands",
        "Water",
        "Clouds and Shadows",
    )
    colormap = (
        "#231F20",
        "#DB5F57",
        "#DB9757",
        "#DBD057",
        "#ADDB57",
        "#75DB57",
        "#7BC47B",
        "#58B158",
        "#D4F6D4",
        "#B0E2B0",
        "#008000",
        "#58B0A7",
        "#995D13",
        "#579BDB",
        "#0062FF",
        "#231F20",
    )
    metadata = {
        "train": {
            "filename": "labeled_train.zip",
            "md5": "2e87d6a218e466dd0566797d7298c7a9",
            "directory": "labeled_train",
        },
        "train-unlabeled": {
            "filename": "unlabeled_train.zip",
            "md5": "1016d724bc494b8c50760ae56bb0585e",
            "directory": "unlabeled_train",
        },
        "val": {"filename": "val.zip", "md5": "6ddd9c0f89d8e74b94ea352d4002073f", "directory": "val"},
    }

    image_root = "BDORTHO"
    dem_root = "RGEALTI"
    target_root = "UrbanAtlas"

    def __init__(self, root="data", split="train", transforms=None, checksum=True):
        """Initialize a new DFC2022 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: if ``split`` is invalid
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in self.metadata
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files()

    def __getitem__(self, index):
        files = self.files[index]
        image = self._load_image(files["image"])
        dem = self._load_image(files["dem"], shape=image.shape[:2])
        image = ops.concatenate([image, dem], axis=-1)

        sample = {"image": image}

        if self.split == "train":
            mask = self._load_target(files["target"])
            sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _load_files(self):
        """Return the paths of the files in the dataset."""
        directory = os.path.join(self.root, self.metadata[self.split]["directory"])
        images = glob.glob(os.path.join(directory, "**", self.image_root, "*.tif"), recursive=True)

        files = []
        for image in sorted(images):
            dem = image.replace(self.image_root, self.dem_root)
            dem = f"{os.path.splitext(dem)[0]}_RGEALTI.tif"

            if self.split == "train":
                target = image.replace(self.image_root, self.target_root)
                target = f"{os.path.splitext(target)[0]}_UA2012.tif"
                files.append({"image": image, "dem": dem, "target": target})
            else:
                files.append({"image": image, "dem": dem})

        return files

    def _load_image(self, path, shape=None):
        """Load a single image, returned channels-last (H, W, C)."""
        with rasterio.open(path) as f:
            array = f.read(out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear)
            array = np.transpose(array, (1, 2, 0))
            return ops.convert_to_tensor(array)

    def _load_target(self, path):
        """Load the target mask for a single image."""
        with rasterio.open(path) as f:
            array = f.read(indexes=1, out_dtype="int32", resampling=Resampling.bilinear)
            return ops.cast(ops.convert_to_tensor(array), "int64")

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        exists = []
        for split_info in self.metadata.values():
            exists.append(os.path.exists(os.path.join(self.root, split_info["directory"])))

        if all(exists):
            return

        # Check if .zip files already exists (if so then extract)
        exists = []
        for split_info in self.metadata.values():
            filepath = os.path.join(self.root, split_info["filename"])
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, split_info["md5"]):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        raise DatasetNotFoundError(self)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt
        from matplotlib import colors

        ncols = 2
        image = ops.convert_to_numpy(sample["image"][:, :, :3]).astype("uint8")

        dem = quantile_normalization(sample["image"][:, :, -1])
        dem = ops.convert_to_numpy(dem)

        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        cmap = colors.ListedColormap(self.colormap)

        if showing_mask:
            mask = ops.convert_to_numpy(sample["mask"])
            ncols += 1
        if showing_prediction:
            pred = ops.convert_to_numpy(sample["prediction"])
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 10, 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dem)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(mask, cmap=cmap, interpolation="none")
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(pred, cmap=cmap, interpolation="none")
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(pred, cmap=cmap, interpolation="none")
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DEM")

            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
