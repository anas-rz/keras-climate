"""Major TOM datasets (ported from torchgeo.datasets.major_tom)."""

import geopandas as gpd
import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset


class MajorTOMEmbeddings(NonGeoDataset):
    """Major TOM Embeddings dataset.

    `Major TOM <https://huggingface.co/Major-TOM>`__ (Terrestrial Observation
    Metaset) is a standard for curating, sharing and combining large-scale EO
    datasets. This data loader provides access to the official embedding
    datasets created using Major TOM Core and several existing foundation
    models.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2412.05600
    """

    def __init__(self, root="data", transforms=None):
        """Initialize a new MajorTOMEmbeddings instance.

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
        """Return the number of data points in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Return an index within the dataset."""
        row = self.data.iloc[index]
        t = pd.Timestamp(row["timestamp"])

        sample = {
            "embedding": ops.convert_to_tensor(np.asarray(row["embedding"])),
            "x": ops.convert_to_tensor(np.asarray(row["centre_lon"])),
            "y": ops.convert_to_tensor(np.asarray(row["centre_lat"])),
            "t": ops.convert_to_tensor(np.asarray(t.timestamp())),
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
            t = pd.Timestamp.fromtimestamp(float(ops.convert_to_numpy(sample["t"])))
            ax.set_title(rf"{y:0.3f}°N, {x:0.3f}°W, {t}")

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        return fig
