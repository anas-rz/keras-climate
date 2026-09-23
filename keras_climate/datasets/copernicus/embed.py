"""Copernicus-Embed dataset (ported from torchgeo.datasets.copernicus.embed)."""

import numpy as np
from keras import ops

from ..errors import DatasetNotFoundError
from ..geo import RasterDataset
from ..utils import download_url


class CopernicusEmbed(RasterDataset):
    """Copernicus-Embed dataset.

    `Copernicus-Embed
    <https://github.com/zhu-xlab/Copernicus-FM/tree/main/Copernicus-Embed-025deg>`__
    is an embedding dataset that gives each 0.25x0.25 grid one embedding vector,
    aggregated over all available modalities from the whole Copernicus-Pretrain dataset
    (721x1440x768, filling empty ocean grids with 0). This dataset can be seen as a
    semantic representation product that integrates various sources of satellite
    observations at an extremely high compression ratio. It also makes it very
    convenient to link Earth's surface to the atmosphere (e.g., as improved static
    variables adding to ERA5), unlocking new possibilities in the development of
    weather/climate foundation models.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849
    """

    filename_glob = "embed_map_*"

    url = "https://hf.co/datasets/torchgeo/copernicus_embed/resolve/435b4a7bdce6f6fdbf4272f9d6e54f2604f35fdb/embed_map_310k.tif"
    sha256 = "38f6fd15b20153f7b3ba2d7b38fb9ef56b751dfcb42e17cc1b02aad1580a59ad"

    def __init__(
        self,
        paths="data",
        crs=None,
        res=None,
        transforms=None,
        cache=True,
        download=False,
        checksum=True,
        time_series=False,
    ):
        """Initialize a new CopernicusEmbed instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: coordinate reference system to warp to (defaults to the CRS
                of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres)
                format. If a single float is provided, it is used for both
                the x and y resolution (defaults to the resolution of the
                first file found)
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)
            time_series: if True, stack data along the time series dimension
                (``T x H x W x C``). If False, merge data into an ``H x W x C``
                mosaic.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

    def _verify(self):
        """Verify the integrity of the dataset."""
        if self.files:
            return

        if self.download:
            download_url(
                self.url, self.paths, sha256=self.sha256 if self.checksum else None
            )
        else:
            raise DatasetNotFoundError(self)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        .. warning::
           Visualizations are generated using PCA on each image
           *individually*, and are thus not comparable across images. The
           plot method is provided for visualization purposes only and
           should not be used to draw conclusions.
        """
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"]).astype(np.float64)
        h, w, c = image.shape
        A = image.reshape(h * w, c)

        # Use PCA to project embeddings from a high-dimensional to a 3D space
        valid = A.sum(axis=1) != 0
        invalid = ~valid

        centered = A[valid] - A[valid].mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        v = vt[:3].T
        b = A @ v

        b -= b[valid].min(axis=0, keepdims=True)
        b /= b[valid].max(axis=0, keepdims=True)
        b[invalid] = 1
        image = b.reshape(h, w, 3)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Embedding")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
