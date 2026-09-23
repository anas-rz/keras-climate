"""South Africa Crop Type Competition Dataset (ported from
torchgeo.datasets.south_africa_crop_type).
"""

import os
import re

import numpy as np
import pandas as pd
import rasterio
from keras import ops
from matplotlib.colors import ListedColormap

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import RasterDataset
from .utils import quantile_normalization, which


class SouthAfricaCropType(RasterDataset):
    """South Africa Crop Type Challenge dataset.

    The `South Africa Crop Type Challenge
    <https://source.coop/radiantearth/south-africa-crops-competition>`__
    dataset includes satellite imagery from Sentinel-1 and Sentinel-2 and labels for
    crop type that were collected by aerial and vehicle survey from May 2017 to March
    2018. Data was provided by the Western Cape Department of Agriculture and is
    available via the Radiant Earth Foundation. For each field id the dataset contains
    time series imagery and a single label mask. Since only the first available
    imagery in July is returned for each field, the dates for S1 and S2 imagery for a
    given field are not guaranteed to be the same. Due to this date mismatch only S1
    or S2 bands may be queried at a time, a mix of both is not supported. Each pixel
    in the label contains an integer field number and crop type class.

    Dataset format:

    * images are 2-band Sentinel 1 and 12-band Sentinel-2 data with a cloud mask
    * masks are tiff images with unique values representing the class and field id.

    Dataset classes:

    0. No Data
    1. Lucerne/Medics
    2. Planted pastures (perennial)
    3. Fallow
    4. Wine grapes
    5. Weeds
    6. Small grain grazing
    7. Wheat
    8. Canola
    9. Rooibos

    If you use this dataset in your research, please cite the following dataset:

    * Western Cape Department of Agriculture, Radiant Earth Foundation (2021)
      "Crop Type Classification Dataset for Western Cape, South Africa",
      Version 1.0, Radiant MLHub, https://doi.org/10.34911/rdnt.j0co8q

    .. note::
       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/ref-south-africa-crops-competition-v1"

    filename_glob = "*_07_*_{}_10m.*"
    filename_regex = r"""
        ^(?P<field_id>\d+)
        _(?P<date>\d{4}_07_\d{2})
        _(?P<band>[BHV\d]+)
        _10m
    """
    date_format = "%Y_%m_%d"
    rgb_bands = ("B04", "B03", "B02")
    s1_bands = ("VH", "VV")
    s2_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    )
    all_bands = s1_bands + s2_bands
    cmap = ListedColormap(
        np.array(
            [
                (0, 0, 0, 255),
                (255, 211, 0, 255),
                (255, 37, 37, 255),
                (0, 168, 226, 255),
                (255, 158, 9, 255),
                (37, 111, 0, 255),
                (255, 255, 0, 255),
                (222, 166, 9, 255),
                (111, 166, 0, 255),
                (0, 175, 73, 255),
            ]
        )
        / 255
    )

    def __init__(
        self,
        paths="data",
        crs=None,
        classes=list(range(10)),
        bands=s2_bands,
        transforms=None,
        download=False,
        time_series=False,
    ):
        """Initialize a new South Africa Crop Type dataset instance.

        Args:
            paths: paths directory where dataset can be found
            crs: coordinate reference system to be used
            classes: crop type classes to be included
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            time_series: if True, stack data along the time series dimension
                [T, H, W, C]. If False, merge data into a [H, W, C] mosaic.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(classes) <= set(range(10)), (
            "Only the following classes are valid: [0-9]."
        )
        assert 0 in classes, "Classes must include the background class: 0"

        self.paths = paths
        self.download = download
        self.filename_glob = self.filename_glob.format(bands[0])

        self._verify()

        super().__init__(
            paths=paths,
            crs=crs,
            bands=bands,
            transforms=transforms,
            time_series=time_series,
        )

        # Map chosen classes to ordinal numbers, all others mapped to background class
        self.ordinal_map = np.zeros(10, dtype=self.dtype)
        self.inverse_map = np.zeros(len(classes), dtype=self.dtype)
        for v, k in enumerate(classes):
            self.ordinal_map[k] = v
            self.inverse_map[v] = k

    def __getitem__(self, index):
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        data_list = []
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        # Loop through matched filepaths and find all unique field ids
        field_ids = []
        # Store date in July for s1 and s2 we want to use for each sample
        imagery_dates = {}

        for filepath in df.filepath:
            filename = os.path.basename(filepath)
            match = re.match(filename_regex, filename)
            if match:
                field_id = match.group("field_id")
                date = match.group("date")
                band = match.group("band")
                band_type = "s1" if band in self.s1_bands else "s2"
                if field_id not in field_ids:
                    field_ids.append(field_id)
                    imagery_dates[field_id] = {"s1": "", "s2": ""}
                if (
                    date.split("_")[1] == "07"
                    and not imagery_dates[field_id][band_type]
                ):
                    imagery_dates[field_id][band_type] = date

        # Create Tensors for each band using stored dates
        paths = self.paths
        for band in self.bands:
            band_type = "s1" if band in self.s1_bands else "s2"
            band_filepaths = []
            for field_id in field_ids:
                date = imagery_dates[field_id][band_type]
                filepath = os.path.join(
                    paths,
                    "train",
                    "imagery",
                    band_type,
                    field_id,
                    date,
                    f"{field_id}_{date}_{band}_10m.tif",
                )
                band_filepaths.append(filepath)
            data_list.append(self._merge_or_stack(band_filepaths, index))
        image = ops.concatenate(data_list, axis=-1)

        # Add labels for each field
        mask_filepaths = []
        for field_id in field_ids:
            file_path = filepath = os.path.join(
                paths, "train", "labels", f"{field_id}.tif"
            )
            mask_filepaths.append(file_path)

        mask = ops.squeeze(self._merge_or_stack(mask_filepaths, index), axis=-1)

        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            "bounds": self._slice_to_tensor(index),
            "image": ops.cast(image, "float32"),
            "mask": ops.cast(mask, "int64"),
            "transform": ops.convert_to_tensor(np.array(list(transform))),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if self.files:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self):
        """Download the dataset."""
        paths = self.paths
        os.makedirs(paths, exist_ok=True)
        azcopy = which("azcopy")
        azcopy("sync", f"{self.url}", paths, "--recursive=true")

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        import matplotlib.pyplot as plt

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = ops.take(sample["image"], rgb_indices, axis=-1)
        image = quantile_normalization(image)

        mask_np = ops.convert_to_numpy(sample["mask"]).astype("int64")
        mask = self.inverse_map[mask_np]
        ncols = 2

        showing_prediction = "prediction" in sample
        if showing_prediction:
            pred_np = ops.convert_to_numpy(sample["prediction"]).astype("int64")
            pred = self.inverse_map[pred_np]
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        kwargs = {"cmap": self.cmap, "vmin": 0, "vmax": 9, "interpolation": "none"}
        axs[0].imshow(ops.convert_to_numpy(image))
        axs[0].axis("off")
        axs[1].imshow(mask, **kwargs)
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_prediction:
            axs[2].imshow(pred, **kwargs)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
