"""Clay Embeddings dataset (ported from torchgeo.datasets.clay)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset


class ClayEmbeddings(NonGeoDataset):
    """Clay Embeddings dataset.

    Supports:

    * `Clay v0 Sentinel embeddings <https://source.coop/clay/clay-model-v0-embeddings>`_
    * `Clay v1.5 NAIP embeddings <https://source.coop/clay/clay-v1-5-naip-2>`_
    * `Clay v1.5 Sentinel embeddings <https://source.coop/clay/lgnd-clay-v1-5-sentinel-2-l2a>`_

    See https://clay-foundation.github.io/model/ for details.
    """

    def __init__(self, root="data", transforms=None):
        """Initialize a new ClayEmbeddings instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        self.transforms = transforms

        try:
            self.data = gpd.read_parquet(root)
        except (FileNotFoundError, ValueError):
            raise DatasetNotFoundError(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        centroid = shapely.centroid(row["geometry"])
        key = "embedding" if "embedding" in row else "embeddings"

        sample = {
            "embedding": ops.convert_to_tensor(np.asarray(row[key])),
            "x": ops.convert_to_tensor(np.asarray(centroid.x)),
            "y": ops.convert_to_tensor(np.asarray(centroid.y)),
        }

        if "date" in row:
            sample["t"] = ops.convert_to_tensor(np.asarray(pd.Timestamp(row["date"]).timestamp()))
        elif "datetime" in row:
            sample["t"] = ops.convert_to_tensor(
                np.asarray(pd.Timestamp(row["datetime"]).timestamp())
            )

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(ops.convert_to_numpy(sample["embedding"]))

        if show_titles:
            x = float(ops.convert_to_numpy(sample["x"]))
            y = float(ops.convert_to_numpy(sample["y"]))
            if "t" in sample:
                t = pd.Timestamp.fromtimestamp(float(ops.convert_to_numpy(sample["t"])))
                ax.set_title(rf"{y:0.3f}°N, {x:0.3f}°W, {t}")

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        return fig
