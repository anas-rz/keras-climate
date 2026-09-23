"""Google Satellite Embedding dataset (ported from torchgeo.datasets.gse)."""

import pathlib

import numpy as np
from keras import ops

from .geo import RasterDataset
from .utils import disambiguate_timestamp


class GoogleSatelliteEmbedding(RasterDataset):
    """Google Satellite Embedding dataset.

    The `Google Satellite Embedding dataset
    <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL>`__
    is a global, analysis-ready collection of learned geospatial embeddings.
    Each 10-meter pixel in this dataset is a 64-dimensional representation, or
    "embedding vector", that encodes temporal trajectories of surface
    conditions at and around that pixel as measured by various Earth
    observation instruments and datasets, over a single calendar year.

    The embeddings are unit-length and do not require any additional
    normalization, and are distributed across the unit sphere, making them
    well-suited for use with clustering algorithms and tree-based
    classifiers.

    The Satellite Embedding dataset was produced by `AlphaEarth Foundations
    <https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/>`__.

    If you use this dataset in your research, please cite the following
    paper:

    * https://arxiv.org/abs/2507.22291
    """

    # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL#bands
    all_bands = tuple(f"A{n:02}" for n in range(64))

    def _filepath_to_timestamp(self, filepath):
        """Extract minimum and maximum timestamps from the filepath.

        Example file paths:

        * GCS/SC: 2024/10N/x086q72fv2f9q1x4a-0000000000-0000000000.tiff
        * HF:     2024/U/1/L/7/471U_587L.tif
        """
        date_format = "%Y"
        for part in pathlib.Path(filepath).parts[::-1]:
            try:
                return disambiguate_timestamp(part, date_format)
            except ValueError:
                pass

        return self.mint, self.maxt

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        .. warning::
           Visualizations are generated using PCA on each image
           *individually*, and are thus not comparable across images. The
           plot method is provided for visualization purposes only and
           should not be used to draw conclusions.
        """
        import matplotlib.pyplot as plt

        h, w, _ = sample["image"].shape
        image = ops.convert_to_numpy(sample["image"])
        A = image.reshape(h * w, -1).astype("float64")

        # Use PCA (via SVD) to project embeddings from 64D to 3D space
        A_centered = A - A.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(A_centered, full_matrices=False)
        v = vt[:3].T
        b = A @ v

        b -= b.min(axis=0, keepdims=True)
        b /= b.max(axis=0, keepdims=True)
        image = b.reshape(h, w, 3)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Embedding")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
