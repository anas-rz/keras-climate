"""Dataset for the Global Biodiversity Information Facility (ported from torchgeo.datasets.gbif)."""

import functools
import glob
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


class GBIF(GeoDataset):
    """Dataset for the Global Biodiversity Information Facility.

    `GBIF <https://www.gbif.org/>`__, the Global Biodiversity Information
    Facility, is an international network and data infrastructure funded by
    the world's governments and aimed at providing anyone, anywhere, open
    access to data about all types of life on Earth.

    This dataset is intended for use with GBIF's
    `occurrence records <https://www.gbif.org/occurrence/search>`_. It may or
    may not work for other GBIF `datasets <https://www.gbif.org/dataset/search>`_.
    Data for a particular species or region of interest can be downloaded
    from the above link.

    If you use a GBIF dataset in your research, please cite it according to:

    * https://www.gbif.org/citation-guidelines
    """

    def __init__(self, root="data"):
        """Initialize a new GBIF dataset instance.

        Args:
            root: root directory where dataset can be found

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root

        files = glob.glob(os.path.join(root, "**.csv"))
        if not files:
            raise DatasetNotFoundError(self)

        # Read tab-delimited CSV file
        usecols = ["decimalLatitude", "decimalLongitude", "day", "month", "year"]
        dtype = {"day": str, "month": str, "year": str}
        df = pd.read_table(files[0], usecols=usecols, dtype=dtype)
        df = df[df["decimalLatitude"].notna()]
        df = df[df["decimalLongitude"].notna()]
        df["day"] = df["day"].str.zfill(2)
        df["month"] = df["month"].str.zfill(2)
        date = df["day"] + " " + df["month"] + " " + df["year"]

        # Convert from pandas DataFrame to geopandas GeoDataFrame
        func = functools.partial(disambiguate_timestamp, format="%d %m %Y")
        index = pd.IntervalIndex.from_tuples(
            date.apply(func).to_list(), closed="both", name="datetime"
        )
        geometry = gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"])
        self.index = GeoDataFrame(index=index, geometry=geometry, crs="EPSG:4326")

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
            raise IndexError(f"index: {index} not found in dataset with bounds: {self.bounds}")

        keypoints = ops.convert_to_tensor(df.get_coordinates().values.astype("float32"))
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
            ax.set_title("GBIF Occurrence Locations by Date")

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
