"""MMEarth Dataset (ported from torchgeo.datasets.mmearth).

.. important::
   Spatial modalities (those with height/width dimensions, e.g. ``sentinel2``,
   ``esa_worldcover``, ``aster``) are returned channels-last (``H x W x C``)
   here, unlike torchgeo's channels-first (``C x H x W``). Non-spatial,
   vector-valued modalities (``era5``, ``biome``, ``eco_region``) are
   returned unchanged, since they have no spatial axes to reorder.
"""

import json
import os

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import lazy_import, quantile_normalization


class MMEarth(NonGeoDataset):
    """MMEarth dataset.

    There are three different versions of the dataset, that vary in image
    size and the number of tiles:

    * MMEarth: 128x128 px, 1.2M tiles, 579 GB
    * MMEarth64: 64x64 px, 1.2M tiles, 162 GB
    * MMEarth100k: 128x128 px, 100K tiles, 48 GB

    The dataset consists of 12 modalities: Aster, Biome, ETH Canopy Height,
    Dynamic World, Ecoregion, ERA5, ESA World Cover, Sentinel-1, Sentinel-2,
    Geolocation, and Date. Additionally, there are three masks available as
    modalities: Sentinel-2 Cloudmask, Sentinel-2 Cloud probability, and
    Sentinel-2 SCL.

    Dataset format:

    * Dataset in single HDF5 file
    * JSON files for band statistics, splits, and tile information

    For additional information, as well as bash scripts to download the
    data, please refer to the
    `official repository <https://github.com/vishalned/MMEarth-data?tab=readme-ov-file#data-download>`_.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.02771

    .. note::

       This dataset requires the following additional library to be
       installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset
    """

    subsets = ("MMEarth", "MMEarth64", "MMEarth100k")

    filenames = {
        "MMEarth": "data_1M_v001",
        "MMEarth64": "data_1M_v001_64",
        "MMEarth100k": "data_100k_v001",
    }

    all_modalities = (
        "aster",
        "biome",
        "canopy_height_eth",
        "dynamic_world",
        "eco_region",
        "era5",
        "esa_worldcover",
        "sentinel1_asc",
        "sentinel1_desc",
        "sentinel2",
        "sentinel2_cloudmask",
        "sentinel2_cloudprod",
        "sentinel2_scl",
    )

    # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/MODALITIES.py#L108C1-L160C2
    all_modality_bands = {
        "sentinel2": [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8A",
            "B8",
            "B9",
            "B10",
            "B11",
            "B12",
        ],
        "sentinel2_cloudmask": ["QA60"],
        "sentinel2_cloudprod": ["MSK_CLDPRB"],
        "sentinel2_scl": ["SCL"],
        "sentinel1_asc": ["VV", "VH", "HH", "HV"],
        "sentinel1_desc": ["VV", "VH", "HH", "HV"],
        "aster": ["b1", "slope"],
        "era5": [
            "prev_temperature_2m",
            "prev_temperature_2m_min",
            "prev_temperature_2m_max",
            "prev_total_precipitation_sum",
            "curr_temperature_2m",
            "curr_temperature_2m_min",
            "curr_temperature_2m_max",
            "curr_total_precipitation_sum",
            "0_temperature_2m_mean",
            "1_temperature_2m_min_min",
            "2_temperature_2m_max_max",
            "3_total_precipitation_sum_sum",
        ],
        "dynamic_world": ["label"],
        "canopy_height_eth": ["height", "std"],
        "lat": ["sin", "cos"],
        "lon": ["sin", "cos"],
        "biome": ["biome"],
        "eco_region": ["eco_region"],
        "month": ["sin_month", "cos_month"],
        "esa_worldcover": ["Map"],
    }

    # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/MODALITIES.py#L36
    no_data_vals = {
        "sentinel2": 0,
        "sentinel2_cloudmask": 65535,
        "sentinel2_cloudprod": 65535,
        "sentinel2_scl": 255,
        "sentinel1_asc": float("-inf"),
        "sentinel1_desc": float("-inf"),
        "aster": float("-inf"),
        "canopy_height_eth": 255,
        "dynamic_world": 0,
        "esa_worldcover": 255,
        "lat": float("-inf"),
        "lon": float("-inf"),
        "month": float("-inf"),
        "era5": float("inf"),
        "biome": 255,
        "eco_region": 65535,
    }

    norm_modes = ("z-score", "min-max")

    modality_category_name = {
        "sentinel1_asc": "image_",
        "sentinel1_desc": "image_",
        "sentinel2": "image_",
        "sentinel2_cloudmask": "mask_",
        "sentinel2_cloudprod": "mask_",
        "sentinel2_scl": "mask_",
        "aster": "image_",
        "era5": "",
        "canopy_height_eth": "image_",
        "dynamic_world": "mask_",
        "esa_worldcover": "mask_",
    }

    #: Modalities with height/width spatial axes (returned channels-last).
    _spatial_modalities = frozenset(
        {
            "aster",
            "canopy_height_eth",
            "dynamic_world",
            "esa_worldcover",
            "sentinel1_asc",
            "sentinel1_desc",
            "sentinel2",
            "sentinel2_cloudmask",
            "sentinel2_cloudprod",
            "sentinel2_scl",
        }
    )

    def __init__(
        self,
        root="data",
        subset="MMEarth",
        modalities=all_modalities,
        modality_bands=None,
        normalization_mode="z-score",
        transforms=None,
    ):
        """Initialize the MMEarth dataset.

        Args:
            root: root directory where dataset can be found
            subset: one of "MMEarth", "MMEarth64", or "MMEarth100k"
            modalities: list of modalities to load
            modality_bands: dictionary of modality bands
            normalization_mode: one of "z-score" or "min-max"
            transforms: a function/transform that takes input sample
                dictionary and returns a transformed version

        Raises:
            AssertionError: if *normalization_mode* or *subset* is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        lazy_import("h5py")

        assert normalization_mode in self.norm_modes, (
            f"Invalid normalization mode: {normalization_mode}, please choose "
            f"from {self.norm_modes}"
        )
        assert subset in self.subsets, (
            f"Invalid dataset version: {subset}, please choose from {self.subsets}"
        )

        self._validate_modalities(modalities)
        self.modalities = modalities
        if modality_bands is None:
            modality_bands = {
                modality: self.all_modality_bands[modality] for modality in modalities
            }
        self._validate_modality_bands(modality_bands)
        self.modality_bands = modality_bands

        self.root = root
        self.subset = subset
        self.normalization_mode = normalization_mode
        self.split = "train"
        self.transforms = transforms

        self.dataset_filename = f"{self.filenames[subset]}.h5"
        self.band_stats_filename = f"{self.filenames[subset]}_band_stats.json"
        self.splits_filename = f"{self.filenames[subset]}_splits.json"
        self.tile_info_filename = f"{self.filenames[subset]}_tile_info.json"

        self._verify()

        self.indices = self._load_indices()
        self.band_stats = self._load_normalization_stats()
        self.tile_info = self._load_tile_info()

    def _verify(self):
        """Verify the dataset."""
        data_dir = os.path.join(self.root, self.filenames[self.subset])

        exists = [
            os.path.exists(os.path.join(data_dir, f))
            for f in [
                self.dataset_filename,
                self.band_stats_filename,
                self.splits_filename,
                self.tile_info_filename,
            ]
        ]
        if not all(exists):
            raise DatasetNotFoundError(self)

    def _load_indices(self):
        """Load the indices for the dataset split."""
        with open(
            os.path.join(self.root, self.filenames[self.subset], self.splits_filename)
        ) as f:
            return json.load(f)[self.split]

    def _load_normalization_stats(self):
        """Load normalization statistics for each band."""
        with open(
            os.path.join(
                self.root, self.filenames[self.subset], self.band_stats_filename
            )
        ) as f:
            return json.load(f)

    def _load_tile_info(self):
        """Load tile information."""
        with open(
            os.path.join(
                self.root, self.filenames[self.subset], self.tile_info_filename
            )
        ) as f:
            return json.load(f)

    def _validate_modalities(self, modalities):
        """Validate list of modalities.

        Raises:
            AssertionError: if ``modalities`` is not a sequence
            ValueError: if an invalid modality name is provided
        """
        from collections.abc import Sequence

        assert isinstance(modalities, Sequence), "'modalities' must be a sequence"
        if not set(modalities) <= set(self.all_modalities):
            raise ValueError(
                f"{set(modalities) - set(self.all_modalities)} is an invalid modality."
            )

    def _validate_modality_bands(self, modality_bands):
        """Validate modality bands.

        Raises:
            AssertionError: if ``modality_bands`` is not a dictionary
            ValueError: if an invalid modality or band name is provided
        """
        assert isinstance(modality_bands, dict), "'modality_bands' must be a dictionary"
        for key, vals in modality_bands.items():
            if key not in self.modalities:
                raise ValueError(f"'{key}' is an invalid modality name.")
            for val in vals:
                if val not in self.all_modality_bands[key]:
                    raise ValueError(
                        f"'{val}' is an invalid band name for modality '{key}'."
                    )

    def __getitem__(self, index):
        """Return a sample from the dataset.

        Normalization is applied to the data with chosen ``normalization_mode``.
        In addition to the modalities, the sample contains the following raw
        metadata: lat, lon, date, tile_id.
        """
        ds_index = self.indices[index]

        sample = self._retrieve_sample(ds_index)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _get_sample_specific_band_names(self, tile_info):
        """Retrieve the sample specific band names."""
        date_str = tile_info["S2_DATE"]
        date_obj = pd.to_datetime(date_str, format="%Y-%m-%d")
        curr_month_str = date_obj.strftime("%Y%m")
        prev_month_obj = date_obj.replace(day=1) - pd.Timedelta(days=1)
        prev_month_str = prev_month_obj.strftime("%Y%m")

        specific_modality_bands = {}
        for modality, bands in self.modality_bands.items():
            if modality == "era5":
                bands = [band.replace(prev_month_str, "prev") for band in bands]
                bands = [band.replace(curr_month_str, "curr") for band in bands]
            specific_modality_bands[modality] = bands

        return specific_modality_bands

    def _get_intersection_dict(self, tile_info):
        """Get intersection of requested and available bands."""
        sample_specific_band_names = self._get_sample_specific_band_names(tile_info)
        intersection_dict = {}
        for modality in self.all_modalities:
            if modality in sample_specific_band_names:
                intersected_list = [
                    band
                    for band in self.all_modality_bands[modality]
                    if band in sample_specific_band_names[modality]
                ]
                if intersected_list:
                    intersection_dict[modality] = intersected_list

        return intersection_dict

    def _retrieve_sample(self, ds_index):
        """Retrieve a sample from the dataset."""
        h5py = lazy_import("h5py")
        sample = {}
        with h5py.File(
            os.path.join(self.root, self.filenames[self.subset], self.dataset_filename),
            "r",
        ) as f:
            name = f["metadata"][ds_index][0].decode("utf-8")
            tile_info = self.tile_info[name]
            intersection_dict = self._get_intersection_dict(tile_info)
            for modality, bands in intersection_dict.items():
                if "sentinel1" in modality:
                    data = f["sentinel1"][ds_index][:]
                else:
                    data = f[modality][ds_index][:]

                tensor = self._preprocess_modality(data, modality, tile_info, bands)
                modality_name = self.modality_category_name.get(modality, "") + modality
                sample[modality_name] = tensor

            sample["lat"] = ops.convert_to_tensor(np.asarray(tile_info["lat"]))
            sample["lon"] = ops.convert_to_tensor(np.asarray(tile_info["lon"]))
            sample["date"] = ops.convert_to_tensor(
                np.asarray(pd.Timestamp(tile_info["S2_DATE"]).timestamp())
            )
            sample["tile_id"] = ops.convert_to_tensor(np.asarray(int(name)))

        return sample

    def _select_indices_for_modality(self, modality, bands):
        """Select band indices for a modality."""
        if modality == "sentinel1_desc":
            indices = [
                self.all_modality_bands["sentinel1_desc"].index(band) + 4
                for band in bands
            ]
        elif modality in ["sentinel2_l1c", "sentinel2_l2a"]:
            indices = [self.all_modality_bands["sentinel2"].index(band) for band in bands]
        else:
            indices = [self.all_modality_bands[modality].index(band) for band in bands]
        return indices

    def _preprocess_modality(self, data, modality, tile_info, bands):
        """Preprocess a single modality.

        Returns a Keras tensor. Spatial modalities (band, height, width) are
        transposed to channels-last (height, width, band) before conversion.
        """
        indices = self._select_indices_for_modality(modality, bands)
        data = data[indices, ...]

        # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/mmearth_dataset.py#L69
        if modality == "dynamic_world":
            data = np.where(data == self.no_data_vals[modality], np.nan, data)
            old_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]
            new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan]
            for old, new in zip(old_values, new_values):
                data = np.where(data == old, new, data)
        elif modality == "esa_worldcover":
            old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255]
            new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255]
            for old, new in zip(old_values, new_values):
                data = np.where(data == old, new, data)
            data = data.astype(np.int64)
        elif modality in [
            "aster",
            "canopy_height_eth",
            "sentinel1_asc",
            "sentinel1_desc",
            "sentinel2",
            "era5",
            "lat",
            "lon",
            "month",
        ]:
            data = data.astype(np.float32)
            if modality == "sentinel2":
                modality_ = (
                    "sentinel2_l2a" if tile_info["S2_type"] == "l2a" else "sentinel2_l1c"
                )
            else:
                modality_ = modality
            data = self._normalize_modality(data, modality_, bands)
            data = np.where(data == self.no_data_vals[modality], np.nan, data)
            data = data.astype(np.float32)
        elif modality in ["biome", "eco_region"]:
            data = data.astype(np.int64)
        elif modality in ["sentinel2_cloudmask", "sentinel2_cloudprod", "sentinel2_scl"]:
            data = data.astype(np.int64)

        # TODO: data might still contain nans, how to handle this?
        if modality in self._spatial_modalities and data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))

        return ops.convert_to_tensor(data)

    def _normalize_modality(self, data, modality, bands):
        """Normalize a single modality."""
        indices = self._select_indices_for_modality(modality, bands)

        if "sentinel1" in modality:
            modality = "sentinel1"

        if self.normalization_mode == "z-score":
            mean = np.array(self.band_stats[modality]["mean"])[indices, ...]
            std = np.array(self.band_stats[modality]["std"])[indices, ...]
            if data.ndim == 3:
                data = (data - mean[:, None, None]) / std[:, None, None]
            else:
                data = (data - mean) / std
        elif self.normalization_mode == "min-max":
            min_val = np.array(self.band_stats[modality]["min"])[indices, ...]
            max_val = np.array(self.band_stats[modality]["max"])[indices, ...]
            if data.ndim == 3:
                data = (data - min_val[:, None, None]) / (
                    max_val[:, None, None] - min_val[:, None, None]
                )
            else:
                data = (data - min_val) / (max_val - min_val)

        return data

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.indices)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset as shown in fig. 2 from https://arxiv.org/pdf/2405.02771."""
        import matplotlib.pyplot as plt

        color_map = {
            "esa_worldcover": {
                0: [0, 100, 0],
                1: [255, 187, 34],
                2: [255, 255, 76],
                3: [240, 150, 255],
                4: [250, 0, 0],
                5: [180, 180, 180],
                6: [240, 240, 240],
                7: [0, 100, 200],
                8: [0, 150, 160],
                9: [0, 207, 117],
                10: [250, 230, 160],
                255: [0, 0, 0],
            },
            "dynamic_world": {
                0: [65, 155, 223],
                1: [57, 125, 73],
                2: [136, 176, 83],
                3: [122, 135, 198],
                4: [228, 150, 53],
                5: [223, 195, 90],
                6: [196, 40, 27],
                7: [165, 155, 143],
                8: [179, 159, 225],
            },
        }

        images = []
        titles = []

        keys_to_plot = [
            "image_sentinel2",
            "image_sentinel1_asc",
            "image_aster",
            "mask_esa_worldcover",
            "mask_dynamic_world",
            "image_canopy_height_eth",
        ]

        for key in keys_to_plot:
            if key not in sample:
                continue
            val = ops.convert_to_numpy(sample[key])
            modalities_name = key.split("_", 1)[1]
            match modalities_name:
                case "sentinel2":
                    norm_img = ops.convert_to_numpy(
                        quantile_normalization(val[:, :, [3, 2, 1]])
                    )
                    images.append(norm_img)
                    titles.append("Sentinel-2 RGB")
                case "esa_worldcover":
                    tensor = val.squeeze(-1)
                    rgb_image = np.zeros((*tensor.shape, 3), dtype=np.uint8)
                    for value, color in color_map[modalities_name].items():
                        rgb_image[tensor == value] = color

                    images.append(rgb_image)
                    titles.append(modalities_name.replace("_", " ").title())
                case "dynamic_world":
                    tensor = val.squeeze(-1)
                    rgb_image = np.zeros((*tensor.shape, 3), dtype=np.uint8)
                    for value, color in color_map[modalities_name].items():
                        rgb_image[tensor == value] = color

                    images.append(rgb_image)
                    titles.append(modalities_name.replace("_", " ").title())
                case _:
                    band_val = val[:, :, 0]
                    norm_img = ops.convert_to_numpy(quantile_normalization(band_val))
                    images.append(norm_img)

                    modalities_name = key.split("_", 1)[1]
                    titles.append(modalities_name.replace("_", " ").title())

        fig, ax = plt.subplots(1, len(images) or 1, figsize=(12, 4))
        if len(images) == 1:
            ax = [ax]

        for i, (image, title) in enumerate(zip(images, titles)):
            ax[i].imshow(image)
            ax[i].axis("off")

            if show_titles:
                title_words = title.split(" ")
                title_word_len = len(title_words)
                if title_word_len > 2:
                    title = (
                        str.join(" ", title_words[:2]) + "\n" + str.join(" ", title_words[2:])
                    )
                ax[i].set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig
