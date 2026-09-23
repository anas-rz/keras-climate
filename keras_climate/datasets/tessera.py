"""Tessera embeddings dataset (ported from torchgeo.datasets.tessera)."""

import numpy as np
from keras import ops

from .geo import RasterDataset


class TesseraEmbeddings(RasterDataset):
    """Tessera embeddings dataset.

    This is a data loader for geospatial embeddings from the `Tessera
    foundation model <https://github.com/ucam-eo/tessera>`__, which
    processes Sentinel-1 and Sentinel-2 satellite imagery to generate
    128-channel representation maps at 10m resolution. These embeddings
    compress a full year of temporal-spectral features into dense
    representations optimized for downstream geospatial analysis tasks.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2506.20380

    .. note::
       The dataset can be downloaded using the `geotessera
       <https://github.com/ucam-eo/geotessera>`__ library. Be sure to use
       ``--format tiff`` to download GeoTIFF files compatible with
       keras_climate.
    """

    filename_glob = "grid_*.tiff"
    filename_regex = r"""
        ^grid
        _(?P<lon>[0-9.-]+)
        _(?P<lat>[0-9.-]+)
        _(?P<date>[0-9]{4})
        .tiff$
    """
    date_format = "%Y"
    all_bands = tuple(map(str, range(128)))

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        .. warning::
           Visualizations are generated using PCA on each image
           *individually*, and are thus not comparable across images. The
           plot method is provided for visualization purposes only and
           should not be used to draw conclusions.
        """
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        h, w, c = image.shape
        A = image.reshape(h * w, c)

        # Use PCA to project embeddings from 128D to 3D space
        A_centered = A - A.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(A_centered, full_matrices=False)
        V = Vt[:3].T
        B = A @ V

        B -= B.min(axis=0, keepdims=True)
        B /= B.max(axis=0, keepdims=True)
        image = B.reshape(h, w, 3)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Embedding")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
