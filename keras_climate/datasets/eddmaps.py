"""Dataset for EDDMapS (ported from torchgeo.datasets.eddmaps)."""

import functools
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from geopandas import GeoDataFrame
from keras import ops

from .errors import DatasetNotFoundError
from .geo import GeoDataset
from .utils import disambiguate_timestamp


class EDDMapS(GeoDataset):
    """Dataset for EDDMapS.

    `EDDMapS <https://www.eddmaps.org/>`__, Early Detection and Distribution
    Mapping System, is a web-based mapping system for documenting invasive
    species and pest distribution. Launched in 2005 by the Center for
    Invasive Species and Ecosystem Health at the University of Georgia, it
    was originally designed as a tool for state Exotic Pest Plant Councils to
    develop more complete distribution data of invasive species.

    EDDMapS query results can be downloaded in CSV, KML, or Shapefile format.
    This dataset currently only supports CSV files.

    If you use an EDDMapS dataset in your research, please cite it like so:

    * EDDMapS. *YEAR*. Early Detection & Distribution Mapping System. The
      University of Georgia - Center for Invasive Species and Ecosystem
      Health. Available online at https://www.eddmaps.org/; last accessed
      *DATE*.
    """

    def __init__(self, root="data"):
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root

        filepath = os.path.join(root, "mappings.csv")
        if not os.path.exists(filepath):
            raise DatasetNotFoundError(self)

        df = pd.read_csv(filepath, usecols=["ObsDate", "Latitude", "Longitude"])
        df = df[df.Latitude.notna()]
        df = df[df.Longitude.notna()]

        func = functools.partial(disambiguate_timestamp, format="%m-%d-%y")
        data = df["ObsDate"].apply(func).to_list()
        index = pd.IntervalIndex.from_tuples(data, closed="both", name="datetime")
        geometry = gpd.points_from_xy(df.Longitude, df.Latitude)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs="EPSG:4326")

    def __getitem__(self, index):
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        keypoints = ops.cast(
            ops.convert_to_tensor(df.get_coordinates().values), "float32"
        )
        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            "bounds": self._slice_to_tensor(index),
            "keypoints": keypoints,
            "transform": ops.convert_to_tensor(np.array(list(transform))),
        }

        return sample

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(ls="--")

        keypoints = ops.convert_to_numpy(sample["keypoints"])
        x = keypoints[:, 0]
        y = keypoints[:, 1]

        ax.scatter(x, y)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if show_titles:
            ax.set_title("EDDMapS Observation Locations by Date")

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
