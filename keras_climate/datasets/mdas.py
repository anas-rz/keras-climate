"""MDAS dataset (ported from torchgeo.datasets.mdas)."""

import os

import numpy as np
import rasterio as rio
from keras import ops
from matplotlib.colors import ListedColormap

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive, extract_archive


class MDAS(NonGeoDataset):
    """MDAS dataset.

    The `MDAS <https://essd.copernicus.org/articles/15/113/2023/>`__
    multimodal dataset is a comprehensive dataset for the city of Augsburg,
    Germany, collected on 7th May 2018. It includes SAR, multispectral,
    hyperspectral, DSM, and GIS data, providing comprehensive options for
    data fusion research.

    Dataset format:

    * 3K_RGB.tif (Shape: (15000, 18000, 4)px, Data Type: uint8)
    * 3K_dsm.tif (Shape: (10000, 12000, 1)px, Data Type: float32)
    * HySpex.tif (Shape: (1364, 1636, 368)px, Data Type: int16)
    * EeteS_EnMAP_2dot2m.tif (Shape: (1364, 1636, 242)px, Data Type: float32)
    * EeteS_EnMAP_10m.tif (Shape: (300, 360, 242)px, Data Type: uint16)
    * EeteS_EnMAP_30m.tif (Shape: (100, 120, 242)px, Data Type: uint16)
    * EeteS_Sentinel_2_10m.tif (Shape: (300, 360, 4)px, Data Type: uint16)
    * Sentinel_2.tif (Shape: (300, 360, 12)px, Data Type: uint16)
    * Sentinel_1.tif (Shape: (300, 360, 2)px, Data Type: float32)
    * osm_buildings.tif (Shape: (1364, 1636, 1)px, Data Type: uint8)
    * osm_landuse.tif (Shape: (1364, 1636, 1)px, Data Type: float64)
    * osm_water.tif (Shape: (1364, 1636, 1)px, Data Type: float64)

    If you use this dataset in your research, please cite the following paper:

    * https://essd.copernicus.org/articles/15/113/2023/
    """

    valid_modalities = (
        "3K_DSM",
        "3K_RGB",
        "HySpex",
        "EeteS_EnMAP_10m",
        "EeteS_EnMAP_30m",
        "EeteS_Sentinel_2_10m",
        "Sentinel_2",
        "Sentinel_1",
        "osm_buildings",
        "osm_landuse",
        "osm_water",
    )
    landuse_class_names = {
        0: "no label",
        1: "forest",
        2: "park",
        3: "residential",
        4: "industrial",
        5: "farm",
        6: "cemetery",
        7: "allotments",
        8: "meadow",
        9: "commercial",
        10: "nature reserve",
        11: "recreation ground",
        12: "retail",
        13: "military",
        14: "quarry",
        15: "orchard",
        16: "scrub",
        17: "grass",
        18: "heath",
    }

    # https://github.com/zhu-xlab/augsburg_Multimodal_Data_Set_MDaS/blob/75c015022b5f688dfc44744f19bcf34bdce786c7/Augsburg_data_4_publication/entire_city/OSM_label/README#L14
    landuse_mapping = {
        -2147483647: 0,
        7201: 1,
        7202: 2,
        7203: 3,
        7204: 4,
        7205: 5,
        7206: 6,
        7207: 7,
        7208: 8,
        7209: 9,
        7210: 10,
        7211: 11,
        7212: 12,
        7213: 13,
        7214: 14,
        7215: 15,
        7217: 16,
        7218: 17,
        7219: 18,
    }

    ds_root_name = "Augsburg_data_4_publication"
    zipfilename = f"{ds_root_name}.zip"
    valid_subareas = ("sub_area_1", "sub_area_2", "sub_area_3")
    url = "https://huggingface.co/datasets/torchgeo/mdas/resolve/860226b74269f1cf1bed8ea3c03f571ae701144c/Augsburg_data_4_publication.zip"
    md5 = "7b63c26e3717cb52c6ba47d215f18d5b"

    enmap_rgb_band_idx = [43, 28, 10]
    sentinel_2_rgb_band_idx = [3, 2, 1]
    hyspex_rgb_band_idx = [100, 50, 10]

    def __init__(
        self,
        root="data",
        subareas=None,
        modalities=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new MDAS dataset instance.

        Args:
            root: root directory where the dataset should be stored.
            subareas: the subareas to load. Options are 'sub_area_1',
                'sub_area_2', 'sub_area_3'.
            modalities: the modalities to load.
            transforms: a function/transform that takes in a dictionary and
                returns a transformed version.
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the integrity of the dataset after download.

        Raises:
            AssertionError: If the subareas or modalities are not valid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        if modalities is None:
            modalities = ["3K_RGB", "HySpex", "Sentinel_2"]
        if subareas is None:
            subareas = ["sub_area_1"]
        self.root = root
        self.download = download
        assert all(sub in self.valid_subareas for sub in subareas), (
            f"Subareas must be one of {self.valid_subareas}"
        )
        self.subareas = subareas
        assert all(mod in self.valid_modalities for mod in modalities), (
            f"Modalities must be one of {self.valid_modalities}"
        )
        self.modalities = modalities
        self.transforms = transforms
        self.checksum = checksum

        self._verify()
        self.files = self._load_files()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.files)

    def _load_files(self):
        """Return the paths of the files in the dataset."""
        files = []
        for subarea in self.subareas:
            subarea_files = {}
            for modality in self.modalities:
                subarea_files[modality] = os.path.join(
                    self.root,
                    self.ds_root_name,
                    subarea,
                    f"{modality}_{self._format_subarea(subarea)}.tif",
                )
            files.append(subarea_files)
        return files

    def _format_subarea(self, subarea):
        """Format the subarea name."""
        parts = subarea.split("_")
        return parts[0] + "_" + parts[1] + parts[2]

    def _load_image(self, path):
        """Load an image from a given path, returned channels-last (H, W, C)."""
        with rio.open(path) as src:
            img = src.read()
            if img.dtype == np.uint16:
                img = img.astype(np.int32)
            if "osm_landuse" in str(path):
                img = np.vectorize(self.landuse_mapping.get)(img)

            img = np.transpose(img, (1, 2, 0))
            return ops.convert_to_tensor(img)

    def __getitem__(self, index):
        """Return the dataset sample at the given index."""
        sample_files = self.files[index]
        sample = {}
        for modality, path in sample_files.items():
            if "osm" in modality:
                sample[f"{modality}_mask"] = ops.cast(self._load_image(path), "int64")
            else:
                sample[f"{modality}_image"] = self._load_image(path)

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        exists = []
        for subarea in self.subareas:
            for modality in self.modalities:
                path = os.path.join(
                    self.root,
                    self.ds_root_name,
                    subarea,
                    f"{modality}_{self._format_subarea(subarea)}.tif",
                )
                exists.append(os.path.exists(path))
        if all(exists):
            return

        if os.path.exists(os.path.join(self.root, self.zipfilename)):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _extract(self):
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.zipfilename), self.root)

    def _download(self):
        """Download the dataset."""
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.zipfilename,
            md5=self.md5 if self.checksum else None,
        )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        ncols = len(sample)
        fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        if ncols == 1:
            axs = [axs]

        for idx, (key, data) in enumerate(sample.items()):
            data = ops.convert_to_numpy(data)
            match key:
                case "3K_RGB_image":
                    img = data[:, :, :3] / 255.0
                    axs[idx].imshow(img)
                case "3K_DSM_image":
                    img = data.squeeze(-1)
                    axs[idx].imshow(img, cmap="gray")
                case "EeteS_EnMAP_10m_image" | "EeteS_EnMAP_30m_image":
                    img = data[:, :, self.enmap_rgb_band_idx] / 10000.0
                    axs[idx].imshow(img)
                case "EeteS_Sentinel_2_10m_image":
                    img = data[:, :, self.sentinel_2_rgb_band_idx] / 10000.0
                    axs[idx].imshow(img)
                case "Sentinel_1_image":
                    img = data[:, :, 0].clip(0, 1)
                    axs[idx].imshow(img)
                case "Sentinel_2_image":
                    img = data[:, :, self.sentinel_2_rgb_band_idx] / 10000.0
                    axs[idx].imshow(img)
                case "HySpex_image":
                    img = data[:, :, self.hyspex_rgb_band_idx] / 15000.0
                    axs[idx].imshow(img)
                case "osm_landuse_mask":
                    img = data.squeeze(-1)
                    cmap = ListedColormap([plt.get_cmap("tab20")(i) for i in range(20)])
                    im = axs[idx].imshow(img, cmap=cmap)
                    cbar = plt.colorbar(im, ax=axs[idx], ticks=range(19))
                    cbar.ax.set_yticklabels(
                        [self.landuse_class_names[i] for i in range(19)]
                    )
                case "osm_buildings_mask":
                    img = data.squeeze(-1)
                    axs[idx].imshow(img, cmap="gray")
                case "osm_water_mask":
                    img = data.squeeze(-1)
                    axs[idx].imshow(img, cmap="Blues")

            axs[idx].axis("off")
            if show_titles:
                axs[idx].set_title(key)

        if suptitle:
            plt.suptitle(suptitle)

        return fig
