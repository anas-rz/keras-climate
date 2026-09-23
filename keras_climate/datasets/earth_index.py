"""Earth Index Embeddings dataset (ported from torchgeo.datasets.earth_index)."""

import geopandas as gpd
import numpy as np
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset


class EarthIndexEmbeddings(NonGeoDataset):
    """Earth Index Embeddings dataset.

    `Earth Index Embeddings <https://source.coop/earthgenome/earthindexembeddings>`__
    are a global embedding product generated from Earth Index v2 Sentinel-2
    mosaics. The embeddings are generated using the SoftCon model from Zhu
    XLabs and result in an embedding of length 384. Each embedding captures a
    320 square meter patch of the Earth, gridded using a MajorTom-based grid.
    Embeddings, their IDs and centroids are encoded in geoparquet.
    """

    def __init__(self, root="data", transforms=None):
        """Initialize a new EarthIndexEmbeddings instance.

        Args:
            root: root directory (or geoparquet file) where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        self.transforms = transforms

        try:
            self.data = gpd.read_parquet(root)
        except (FileNotFoundError, ValueError, OSError):
            raise DatasetNotFoundError(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        sample = {
            "embedding": ops.convert_to_tensor(
                np.asarray(row["embedding"], dtype="float32")
            ),
            "x": ops.convert_to_tensor(np.array(row["geometry"].x, dtype="float32")),
            "y": ops.convert_to_tensor(np.array(row["geometry"].y, dtype="float32")),
        }

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
            ax.set_title(rf"{y:0.3f}°N, {x:0.3f}°W")

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        return fig
