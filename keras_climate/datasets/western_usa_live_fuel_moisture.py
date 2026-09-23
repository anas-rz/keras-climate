"""Western USA Live Fuel Moisture dataset (ported from torchgeo.datasets.western_usa_live_fuel_moisture)."""

import glob
import json
import os

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import which


class WesternUSALiveFuelMoisture(NonGeoDataset):
    """Western USA Live Fuel Moisture dataset.

    This tabular style dataset contains fuel moisture (mass of water in
    vegetation) and remotely sensed variables in the western United States.
    It contains 2615 datapoints and 138 variables. For more details see the
    `dataset page <https://source.coop/stanford/sar-moisture-conent>`_.

    Dataset Format:

    * .geojson file for each datapoint

    Dataset Features:

    * 138 remote sensing derived variables, some with a time dependency
    * 2615 datapoints with regression target of predicting fuel moisture

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1016/j.rse.2020.111797

    .. note::

       This dataset requires the following additional library to be
       installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to
         download the dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/su-sar-moisture-content"

    label_name = "percent(t)"

    all_variable_names = (
        "slope(t)",
        "elevation(t)",
        "canopy_height(t)",
        "forest_cover(t)",
        "silt(t)",
        "sand(t)",
        "clay(t)",
        "vv(t)",
        "vh(t)",
        "red(t)",
        "green(t)",
        "blue(t)",
        "swir(t)",
        "nir(t)",
        "ndvi(t)",
        "ndwi(t)",
        "nirv(t)",
        "vv_red(t)",
        "vv_green(t)",
        "vv_blue(t)",
        "vv_swir(t)",
        "vv_nir(t)",
        "vv_ndvi(t)",
        "vv_ndwi(t)",
        "vv_nirv(t)",
        "vh_red(t)",
        "vh_green(t)",
        "vh_blue(t)",
        "vh_swir(t)",
        "vh_nir(t)",
        "vh_ndvi(t)",
        "vh_ndwi(t)",
        "vh_nirv(t)",
        "vh_vv(t)",
        "slope(t-1)",
        "elevation(t-1)",
        "canopy_height(t-1)",
        "forest_cover(t-1)",
        "silt(t-1)",
        "sand(t-1)",
        "clay(t-1)",
        "vv(t-1)",
        "vh(t-1)",
        "red(t-1)",
        "green(t-1)",
        "blue(t-1)",
        "swir(t-1)",
        "nir(t-1)",
        "ndvi(t-1)",
        "ndwi(t-1)",
        "nirv(t-1)",
        "vv_red(t-1)",
        "vv_green(t-1)",
        "vv_blue(t-1)",
        "vv_swir(t-1)",
        "vv_nir(t-1)",
        "vv_ndvi(t-1)",
        "vv_ndwi(t-1)",
        "vv_nirv(t-1)",
        "vh_red(t-1)",
        "vh_green(t-1)",
        "vh_blue(t-1)",
        "vh_swir(t-1)",
        "vh_nir(t-1)",
        "vh_ndvi(t-1)",
        "vh_ndwi(t-1)",
        "vh_nirv(t-1)",
        "vh_vv(t-1)",
        "slope(t-2)",
        "elevation(t-2)",
        "canopy_height(t-2)",
        "forest_cover(t-2)",
        "silt(t-2)",
        "sand(t-2)",
        "clay(t-2)",
        "vv(t-2)",
        "vh(t-2)",
        "red(t-2)",
        "green(t-2)",
        "blue(t-2)",
        "swir(t-2)",
        "nir(t-2)",
        "ndvi(t-2)",
        "ndwi(t-2)",
        "nirv(t-2)",
        "vv_red(t-2)",
        "vv_green(t-2)",
        "vv_blue(t-2)",
        "vv_swir(t-2)",
        "vv_nir(t-2)",
        "vv_ndvi(t-2)",
        "vv_ndwi(t-2)",
        "vv_nirv(t-2)",
        "vh_red(t-2)",
        "vh_green(t-2)",
        "vh_blue(t-2)",
        "vh_swir(t-2)",
        "vh_nir(t-2)",
        "vh_ndvi(t-2)",
        "vh_ndwi(t-2)",
        "vh_nirv(t-2)",
        "vh_vv(t-2)",
        "slope(t-3)",
        "elevation(t-3)",
        "canopy_height(t-3)",
        "forest_cover(t-3)",
        "silt(t-3)",
        "sand(t-3)",
        "clay(t-3)",
        "vv(t-3)",
        "vh(t-3)",
        "red(t-3)",
        "green(t-3)",
        "blue(t-3)",
        "swir(t-3)",
        "nir(t-3)",
        "ndvi(t-3)",
        "ndwi(t-3)",
        "nirv(t-3)",
        "vv_red(t-3)",
        "vv_green(t-3)",
        "vv_blue(t-3)",
        "vv_swir(t-3)",
        "vv_nir(t-3)",
        "vv_ndvi(t-3)",
        "vv_ndwi(t-3)",
        "vv_nirv(t-3)",
        "vh_red(t-3)",
        "vh_green(t-3)",
        "vh_blue(t-3)",
        "vh_swir(t-3)",
        "vh_nir(t-3)",
        "vh_ndvi(t-3)",
        "vh_ndwi(t-3)",
        "vh_nirv(t-3)",
        "vh_vv(t-3)",
        "lat",
        "lon",
    )

    def __init__(self, root="data", input_features=all_variable_names, transforms=None, download=False):
        """Initialize a new Western USA Live Fuel Moisture dataset instance.

        Args:
            root: root directory where dataset can be found
            input_features: which input features to include
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory

        Raises:
            AssertionError: if ``input_features`` contains invalid variable
                names
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert set(input_features) <= set(self.all_variable_names)

        self.root = root
        self.input_features = input_features
        self.transforms = transforms
        self.download = download

        self._verify()

        self.dataframe = self._load_data()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        data = self.dataframe.iloc[index, :]

        sample = {
            "input": ops.convert_to_tensor(
                data.drop([self.label_name]).values.astype("float32")
            ),
            "label": ops.convert_to_tensor(np.array(data[self.label_name], dtype="float32")),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_data(self):
        """Load data from individual files into a pandas dataframe."""
        data_rows = []
        for path in sorted(self.files):
            with open(path) as f:
                content = json.load(f)
                data_dict = content["properties"]
                data_dict["lon"] = content["geometry"]["coordinates"][0]
                data_dict["lat"] = content["geometry"]["coordinates"][1]
                data_rows.append(data_dict)

        df = pd.DataFrame(data_rows)
        df = df[[*self.input_features, self.label_name]]
        return df

    def _verify(self):
        """Verify the integrity of the dataset."""
        file_glob = os.path.join(self.root, "**", "feature_*.geojson")
        self.files = glob.glob(file_glob, recursive=True)
        if self.files:
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self.files = glob.glob(file_glob, recursive=True)

    def _download(self):
        """Download the dataset and extract it."""
        os.makedirs(self.root, exist_ok=True)
        azcopy = which("azcopy")
        azcopy("sync", self.url, self.root, "--recursive=true")

    def plot(self, sample, show_titles=True, suptitle=None, variables_to_plot=None):
        """Plot a time series visualization of the LFMC sample."""
        import matplotlib.pyplot as plt

        if not variables_to_plot:
            variables_to_plot = [
                "slope",
                "elevation",
                "canopy_height",
                "forest_cover",
                "silt",
                "sand",
                "clay",
                "vv",
                "vh",
                "red",
                "green",
                "blue",
                "swir",
                "nir",
                "ndvi",
                "ndwi",
                "nirv",
                "vv_red",
                "vv_green",
                "vv_blue",
                "vv_swir",
                "vv_nir",
                "vv_ndvi",
                "vv_ndwi",
                "vv_nirv",
                "vh_red",
                "vh_green",
                "vh_blue",
                "vh_swir",
                "vh_nir",
                "vh_ndvi",
                "vh_ndwi",
                "vh_nirv",
                "vh_vv",
            ]

        input_data = ops.convert_to_numpy(sample["input"])

        time_labels = ["t", "t-1", "t-2", "t-3"]

        fig, axs = plt.subplots(
            len(variables_to_plot), 1, figsize=(6, 1.5 * len(variables_to_plot)), sharex=True
        )

        if len(variables_to_plot) == 1:
            axs = [axs]

        for i, var_base_name in enumerate(variables_to_plot):
            values = []

            for t_label in time_labels:
                full_var_name = f"{var_base_name}({t_label})"
                var_position = self.all_variable_names.index(full_var_name)
                values.append(input_data[var_position])

            axs[i].plot(range(len(time_labels)), values, "o-")
            axs[i].grid(True, alpha=0.3)

            if show_titles:
                axs[i].set_title(f"{var_base_name.upper()}")

        axs[-1].set_xticks(range(len(time_labels)))
        axs[-1].set_xticklabels(time_labels)

        lon = input_data[-2]
        lat = input_data[-1]
        lfmc_value = float(ops.convert_to_numpy(sample["label"]))

        axs[-1].text(
            x=0.5,
            y=-0.7,
            s=f"Live Fuel Moisture Content\nat {lon:.4f}, {lat:.4f}: {lfmc_value:.2f}%",
            ha="center",
            transform=axs[-1].transAxes,
        )

        if suptitle is not None:
            fig.suptitle(t=suptitle, y=1.6, transform=axs[0].transAxes)

        plt.tight_layout()

        return fig
