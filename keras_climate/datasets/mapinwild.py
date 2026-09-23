"""MapInWild dataset (ported from torchgeo.datasets.mapinwild)."""

import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_url, extract_archive, quantile_normalization


class MapInWild(NonGeoDataset):
    """MapInWild dataset.

    The `MapInWild <https://ieeexplore.ieee.org/document/10089830>`__ dataset
    is curated for the task of wilderness mapping on a pixel-level.
    MapInWild is a multi-modal dataset and comprises various geodata
    acquired and formed from different RS sensors over 1018 locations:
    dual-pol Sentinel-1, four-season Sentinel-2 with 10 bands, ESA
    WorldCover map, and Visible Infrared Imaging Radiometer Suite NightTime
    Day/Night band. The dataset consists of 8144 images with the shape of
    1920 x 1920 pixels. The images are weakly annotated from the World
    Database of Protected Areas (WDPA).

    If you use this dataset in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/10089830
    """

    url = "https://hf.co/datasets/burakekim/mapinwild/resolve/d963778e31e7e0ed2329c0f4cbe493be532f0e71/"

    modality_urls = {
        "esa_wc": {"esa_wc/ESA_WC.zip"},
        "viirs": {"viirs/VIIRS.zip"},
        "mask": {"mask/mask.zip"},
        "s1": {"s1/s1_part1.zip", "s1/s1_part2.zip"},
        "s2_temporal_subset": {
            "s2_temporal_subset/s2_temporal_subset_part1.zip",
            "s2_temporal_subset/s2_temporal_subset_part2.zip",
        },
        "s2_autumn": {"s2_autumn/s2_autumn_part1.zip", "s2_autumn/s2_autumn_part2.zip"},
        "s2_spring": {"s2_spring/s2_spring_part1.zip", "s2_spring/s2_spring_part2.zip"},
        "s2_summer": {"s2_summer/s2_summer_part1.zip", "s2_summer/s2_summer_part2.zip"},
        "s2_winter": {"s2_winter/s2_winter_part1.zip", "s2_winter/s2_winter_part2.zip"},
        "split_IDs": {"split_IDs/split_IDs.csv"},
    }

    sha256s = {
        "ESA_WC.zip": "2705d4b37d6fef5941fe28c3e0897218972374bc821df30d16fe8c149dd65c21",
        "VIIRS.zip": "9e629ca6c7be148bffbbb7468b0b2df541c8da7f629c5bfe4b32bc3f781b45f1",
        "mask.zip": "0d41675fa4b90c2f6400a802e3e0c15a6c2035c7530e3e183bfc9336c5acf458",
        "s1_part1.zip": "304287a8356d03cb4a30e2c26d569aa1e1b69a00539f977f701955c17b72a4d1",
        "s1_part2.zip": "cc00d739ed8580ca1f3865b9eb7938b2f48ee0d2f7db39b1590902157818e889",
        "s2_temporal_subset_part1.zip": "d2e1e1d9c821e90a8df3371502c3fa555dd131b6261e9843077f08fa96c418b3",
        "s2_temporal_subset_part2.zip": "61f48e485bc3f2c5e4a8cf681a681470979723242e9cd46468994d0637627af4",
        "s2_autumn_part1.zip": "119598477923e8c3a71ece56c3e0d85438574c7b70a9c892ede6d9b1e7e89dd8",
        "s2_autumn_part2.zip": "31cf7d1d033250b41e7db23b69ceb5f41481c2d10499707eb52c3decb15a2f74",
        "s2_spring_part1.zip": "00f5fe8e7cfa982b166d1d83ef5e2d16ae4d89ff60046296193c3e5cbd5227fb",
        "s2_spring_part2.zip": "1f2fe86dd908d8c4b41a6478855cba27a9b0ddf46b97d7cf18d239c3be49f2f2",
        "s2_summer_part1.zip": "dfb757efbcc5791c6aaaa1e7b62de3f7bbf51f651b4c9ae94f4d84b771ac65aa",
        "s2_summer_part2.zip": "7241882579d5be88364e1ed2bc9fca427a1aef4192c39b037d45af88972d0e06",
        "s2_winter_part1.zip": "8876d54eddfe708325d94c5efd89278fcc4ad78b7b992bb73f6a9084d0a9a93a",
        "s2_winter_part2.zip": "725ff6d7b2b1ff6bfe7c3e41ccb64e1b878a4d8ba6f5baf224d70978b1696396",
        "split_IDs.csv": "c8b2c5f343fc592b9c50e5be6a24ffa06970313a92a20e16c7114ffc70fcb0fe",
    }

    mask_cmap = {1: (0, 153, 0), 0: (255, 255, 255)}

    wc_cmap = {
        10: (0, 160, 0),
        20: (150, 100, 0),
        30: (255, 180, 0),
        40: (255, 255, 100),
        50: (195, 20, 0),
        60: (255, 245, 215),
        70: (255, 255, 255),
        80: (0, 70, 200),
        90: (0, 220, 130),
        95: (0, 150, 120),
        100: (255, 235, 175),
    }

    def __init__(
        self,
        root="data",
        modality=None,
        split="train",
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new MapInWild dataset instance.

        Args:
            root: root directory where dataset can be found
            modality: the modality to download. Choose from: "mask", "esa_wc",
                "viirs", "s1", "s2_temporal_subset", "s2_[season]".
            split: one of "train", "validation", or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        if modality is None:
            modality = ["mask", "esa_wc", "viirs", "s2_summer"]
        assert split in ["train", "validation", "test"]

        self.checksum = checksum
        self.root = root
        self.transforms = transforms
        self.modality = modality
        self.download = download

        modality.append("split_IDs")
        for mode in modality:
            for modality_link in self.modality_urls[mode]:
                modality_url = os.path.join(self.url, modality_link)
                self._verify(
                    url=modality_url, sha256=self.sha256s[os.path.split(modality_link)[-1]]
                )

            if (
                download
                and mode not in os.listdir(self.root)
                and len(self.modality_urls[mode]) == 2
            ):
                self._merge_parts(mode)

        # Masks will be loaded separately in __getitem__
        if "mask" in self.modality:
            self.modality.remove("mask")

        if "split_IDs" in self.modality:
            self.modality.remove("split_IDs")

        if os.path.exists(os.path.join(self.root, "split_IDs.csv")):
            split_dataframe = pd.read_csv(os.path.join(self.root, "split_IDs.csv"))
            ids = split_dataframe[split].dropna().values.tolist()
            self.ids = list(map(int, ids))

    def __getitem__(self, index):
        """Return an index within the dataset."""
        list_modalities = []
        id = self.ids[index]

        mask = self._load_raster(id, "mask")
        mask = np.where(mask != 0, 1, mask)

        for mode in self.modality:
            mode = mode.upper() if mode in ["esa_wc", "viirs"] else mode
            data = self._load_raster(id, mode)
            list_modalities.append(data)

        image = np.concatenate(list_modalities, axis=-1)

        sample = {
            "image": ops.convert_to_tensor(image),
            "mask": ops.cast(ops.convert_to_tensor(np.squeeze(mask, axis=-1)), "int64"),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.ids)

    def _load_raster(self, filename, source):
        """Load a single raster image or target, returned channels-last (H, W, C)."""
        with rasterio.open(os.path.join(self.root, source, f"{filename}.tif")) as f:
            array = f.read()
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            array = np.transpose(array, (1, 2, 0)).astype(np.float32)
            return array

    def _verify(self, url, sha256=None):
        """Verify the integrity of the dataset."""
        modality_folder_name = url.split("/")[-1]
        mod_fold_no_ext = modality_folder_name.split(".")[0]
        modality_path = os.path.join(self.root, mod_fold_no_ext)
        split_path = os.path.join(self.root, modality_folder_name)
        if mod_fold_no_ext == "split_IDs":
            modality_path = split_path

        if os.path.exists(modality_path):
            return

        filepath = os.path.join(self.root, url.split("/")[-1])
        if os.path.isfile(filepath) and filepath.endswith(".zip"):
            if self.checksum and not check_integrity(filepath, sha256=sha256):
                raise RuntimeError("Dataset found, but corrupted.")
            self._extract(url)
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download(url, sha256)
        if not url.endswith(".csv"):
            self._extract(url)

    def _download(self, url, sha256):
        """Downloads a modality."""
        download_url(
            url, self.root, filename=os.path.split(url)[1], sha256=sha256 if self.checksum else None
        )

    def _extract(self, path):
        """Extracts a modality."""
        filepath = os.path.join(self.root, os.path.split(path)[1])
        extract_archive(filepath)

    def _merge_parts(self, modality):
        """Merge the modalities that are downloaded and extracted in two parts."""
        modality_folder = os.path.join(self.root, modality)
        os.makedirs(modality_folder, exist_ok=True)

        source_folders = [
            os.path.join(self.root, modality + "_part1"),
            os.path.join(self.root, modality + "_part2"),
        ]

        for source_folder in source_folders:
            for file_name in os.listdir(source_folder):
                source = os.path.join(source_folder, file_name)
                destination = os.path.join(modality_folder, file_name)
                if os.path.isfile(source):
                    shutil.copy(source, destination)

    def _convert_to_color(self, arr_2d, cmap):
        """Numeric labels to RGB-color encoding."""
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in cmap.items():
            m = arr_2d == c
            arr_3d[m] = i
        return arr_3d

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        modality_channels = defaultdict(lambda: 10, {"viirs": 1, "esa_wc": 1, "s1": 2})

        start_idx = 0
        split_images = {}

        image = ops.convert_to_numpy(sample["image"])
        for modality in self.modality:
            end_idx = start_idx + modality_channels[modality]
            split_images[modality] = image[:, :, start_idx:end_idx]
            start_idx = end_idx

        mask = ops.convert_to_numpy(sample["mask"]).squeeze()
        color_mask = self._convert_to_color(mask, cmap=self.mask_cmap)

        num_subplots = len(split_images) + 1
        showing_predictions = "prediction" in sample
        if showing_predictions:
            num_subplots += 1

        fig, axs = plt.subplots(1, num_subplots, figsize=(num_subplots * 4, 5))

        for i, (modality, img) in enumerate(split_images.items()):
            ax = axs[i]
            img = img.squeeze()
            if modality.startswith("s2"):
                img = img[:, :, [4, 3, 2]]
            if modality == "esa_wc":
                img = self._convert_to_color(img, cmap=self.wc_cmap)
            if modality == "s1":
                img = img[:, :, 0]

            if modality != "esa_wc":
                img = ops.convert_to_numpy(quantile_normalization(img))

            ax.imshow(img)
            if show_titles:
                ax.set_title(modality)
            ax.axis("off")

        axs[len(split_images)].imshow(color_mask)
        if show_titles:
            axs[len(split_images)].set_title("Annotation")
        axs[len(split_images)].axis("off")

        if showing_predictions:
            prediction = ops.convert_to_numpy(sample["prediction"]).squeeze()
            color_predictions = self._convert_to_color(prediction, cmap=self.mask_cmap)
            axs[-1].imshow(color_predictions, vmin=0, vmax=1, interpolation="none")
            if show_titles:
                axs[-1].set_title("Prediction")
            axs[-1].axis("off")

        plt.tight_layout()
        return fig
