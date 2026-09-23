"""EarthEmbeddings dataset (ported from torchgeo.datasets.earth_embeddings)."""

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset


class EarthEmbeddings(NonGeoDataset):
    """EarthEmbeddings dataset.

    `EarthEmbeddings <https://huggingface.co/datasets/ML4Sustain/EarthEmbeddings>`__
    are pre-computed embeddings of uniformly sampled MajorTOM-Core-S2L2A imagery
    using SatCLIP, FarSLIP, DINOv2, SigLIP models. These embeddings power the
    EarthEmbeddingExplorer application, which allows users to search for
    satellite images using text queries, image uploads, or geographic locations.
    """

    def __init__(self, root="data", transforms=None):
        """Initialize a new EarthEmbeddings instance.

        Args:
            root: root directory (or parquet file) where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        self.transforms = transforms

        try:
            self.data = pd.read_parquet(root)
        except (FileNotFoundError, ValueError, OSError):
            raise DatasetNotFoundError(self)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        t = pd.Timestamp(row["timestamp"])

        sample = {
            "embedding": ops.convert_to_tensor(
                np.asarray(row["embedding"], dtype="float32")
            ),
            "x": ops.convert_to_tensor(np.array(row["centre_lon"], dtype="float32")),
            "y": ops.convert_to_tensor(np.array(row["centre_lat"], dtype="float32")),
            "t": ops.convert_to_tensor(np.array(t.timestamp(), dtype="float64")),
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
