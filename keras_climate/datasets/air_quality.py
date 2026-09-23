"""Air Quality dataset (ported from torchgeo.datasets.air_quality)."""

import math
import pathlib

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset


class AirQuality(NonGeoDataset):
    """Air Quality dataset.

    The `Air Quality dataset <https://archive.ics.uci.edu/dataset/360/air+quality>`_
    from the UCI Machine Learning Repository is a multivariate time series
    dataset containing air quality measurements from an Italian city.

    Dataset Format:

    * .csv file containing date, time and air quality measurements

    Dataset Features:

    * hourly averaged sensor responses and reference analyzer ground truth
      over one year (2004-2005)
    * contains missing features, gap filled using linear interpolation

    .. note:: There are actually two different versions of this dataset with
       major formatting differences, including comma-delimited vs.
       semicolon-delimited, empty rows and columns, and differences in
       datetime formatting. This dataset currently only supports the
       comma-delimited version.

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1016/J.SNB.2007.09.060
    """

    url = "https://archive.ics.uci.edu/static/public/360/data.csv"
    data_file_name = "data.csv"

    def __init__(
        self,
        root="data",
        *,
        input_steps=3,
        target_steps=1,
        input_features=None,
        target_features=None,
        download=False,
    ):
        """Initialize a new Dataset instance.

        Args:
            root: Root directory where dataset can be found.
            input_steps: Number of input time steps to use.
            target_steps: Number of target time steps to use.
            input_features: List of input features to load (uses all
                features by default).
            target_features: List of target features to load (uses all
                features by default).
            download: If True, download dataset and store it in the root
                directory.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.root = pathlib.Path(root)
        self.download = download
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.input_features = input_features
        self.target_features = target_features
        self._load_data()

    def __len__(self):
        return len(self.input_data) - self.input_steps - self.target_steps + 1

    def __getitem__(self, index):
        input = self.input_data.iloc[index : index + self.input_steps]
        target = self.target_data.iloc[
            index + self.input_steps : index + self.input_steps + self.target_steps
        ]

        return {
            "input": ops.convert_to_tensor(input.values.astype("float32")),
            "target": ops.convert_to_tensor(target.values.astype("float32")),
        }

    def _parse_datetime(self, data):
        if {"Date", "Time"} <= set(data.columns):
            dt = pd.to_datetime(data["Date"] + " " + data["Time"]).dt
            doy = 2 * np.pi * dt.dayofyear / 365.25
            hod = 2 * np.pi * dt.hour / 24
            data["sin(DOY)"] = np.sin(doy)
            data["cos(DOY)"] = np.cos(doy)
            data["sin(HOD)"] = np.sin(hod)
            data["cos(HOD)"] = np.cos(hod)

        data.drop(columns=["Date", "Time"], inplace=True, errors="ignore")

    def _load_data(self):
        filepath = self.root / self.data_file_name
        if filepath.is_file():
            pass
        elif self.download:
            filepath = self.url
        else:
            raise DatasetNotFoundError(self)

        # Load twice in case target_features is not a subset of input_features
        kwargs = {"na_values": -200}
        self.input_data = pd.read_csv(filepath, usecols=self.input_features, **kwargs)
        self.target_data = pd.read_csv(filepath, usecols=self.target_features, **kwargs)

        # Encode cyclic features
        self._parse_datetime(self.input_data)
        self._parse_datetime(self.target_data)

        # Interpolate missing values using linear interpolation
        self.input_data.interpolate(inplace=True)
        self.target_data.interpolate(inplace=True)

    def plot(self, sample, show_titles=True, suptitle=None, features=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        ylabel = {
            "CO(GT)": "CO (mg/m$^3$)",
            "PT08.S1(CO)": "CO",
            "NMHC(GT)": "NMHC (μg/m$^3$)",
            "C6H6(GT)": "C$_6$H$_6$ (μg/m$^3$)",
            "PT08.S2(NMHC)": "NHMC",
            "NOx(GT)": "NO$_x$ (ppb)",
            "PT08.S3(NOx)": "NO$_x$",
            "NO2(GT)": "NO$_2$ (μg/m$^3$)",
            "PT08.S4(NO2)": "NO$_2$",
            "PT08.S5(O3)": "O$_3$",
            "T": "Temperature (°C)",
            "RH": "Relative Humidity (%)",
            "AH": "Absolute Humidity",
        }

        sample_input = ops.convert_to_numpy(sample["input"])
        sample_target = ops.convert_to_numpy(sample["target"])

        input_steps = range(len(sample_input))
        target_steps = range(len(sample_input), len(sample_input) + len(sample_target))

        features = features or self.target_data.columns
        n_features = len(features)

        ncols = math.ceil(math.sqrt(n_features))
        nrows = math.ceil(n_features / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False
        )
        axes = axes.ravel()

        for ax, feature in zip(axes, features):
            if show_titles:
                ax.set_title(feature)

            if feature in self.input_data:
                idx = self.input_data.columns.get_loc(feature)
                data = sample_input[:, idx]
                ax.plot(input_steps, data, label="Input", marker="o")

            if feature in self.target_data:
                idx = self.target_data.columns.get_loc(feature)
                data = sample_target[:, idx]
                ax.plot(target_steps, data, label="Target", marker="x")

                if "prediction" in sample:
                    data = ops.convert_to_numpy(sample["prediction"])[:, idx]
                    ax.plot(target_steps, data, label="Prediction", marker="^")

            ax.legend()
            if feature in ylabel:
                ax.set_ylabel(ylabel[feature])

        for ax in axes[n_features:]:
            ax.set_visible(False)

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        return fig
