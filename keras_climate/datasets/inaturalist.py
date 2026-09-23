"""Dataset for iNaturalist (ported from torchgeo.datasets.inaturalist)."""

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


class INaturalist(GeoDataset):
    """Dataset for iNaturalist.

    `iNaturalist <https://www.inaturalist.org/>`__ is a joint initiative of the
    California Academy of Sciences and the National Geographic Society. It
    allows citizen scientists to upload observations of organisms that can be
    downloaded by scientists and researchers.

    If you use an iNaturalist dataset in your research, please cite it
    according to:

    * https://help.inaturalist.org/en/support/solutions/articles/151000170344-how-should-i-cite-inaturalist-
    """

    def __init__(self, root="data"):
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        super().__init__()

        self.root = root

        files = glob.glob(os.path.join(root, "**.csv"))
        if not files:
            raise DatasetNotFoundError(self)

        # Read CSV file
        usecols = ["observed_on", "time_observed_at", "latitude", "longitude"]
        df = pd.read_csv(files[0], header=0, usecols=usecols)
        df = df[df.latitude.notna()]
        df = df[df.longitude.notna()]

        # Convert from pandas DataFrame to geopandas GeoDataFrame
        func = functools.partial(disambiguate_timestamp, format="%Y-%m-%d %H:%M:%S %z")
        time = df.time_observed_at.apply(func)
        func = functools.partial(disambiguate_timestamp, format="%Y-%m-%d")
        date = df.observed_on.apply(func)
        time[time.isnull()] = date[time.isnull()]
        times = time.to_list()
        index = pd.IntervalIndex.from_tuples(times, closed="both", name="datetime")
        geometry = gpd.points_from_xy(df.longitude, df.latitude)
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
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )

        keypoints = ops.convert_to_tensor(
            df.get_coordinates().values.astype("float32")
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

        # Extract coordinates
        keypoints = ops.convert_to_numpy(sample["keypoints"])
        x = keypoints[:, 0]
        y = keypoints[:, 1]

        # Plot the points
        ax.scatter(x, y)

        # Set labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Add titles if requested
        if show_titles:
            ax.set_title("iNaturalist Dataset Plot")

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
