"""Aster Global Digital Elevation Model dataset (ported from
torchgeo.datasets.astergdem).
"""

from .errors import DatasetNotFoundError
from .geo import RasterDataset


class AsterGDEM(RasterDataset):
    """Aster Global Digital Elevation Model Dataset.

    The `Aster Global Digital Elevation Model
    <https://www.earthdata.nasa.gov/data/catalog/lpcloud-astgtm-003>`_
    dataset is a Digital Elevation Model (DEM) on a global scale. The
    dataset can be downloaded from the `Earth Data website
    <https://search.earthdata.nasa.gov/search/>`_ after making an account.

    Dataset features:

    * DEMs at 30 m per pixel spatial resolution (3601x3601 px)
    * data collected from the `Aster
      <https://terra.nasa.gov/about/terra-instruments/aster>`_ instrument

    Dataset format:

    * DEMs are single-channel tif files
    """

    is_image = False
    filename_glob = "ASTGTMV003_*_dem*"
    filename_regex = r"""
        (?P<name>[ASTGTMV003]{10})
        _(?P<id>[A-Z0-9]{7})
        _(?P<data>[a-z]{3})*
    """

    def __init__(
        self,
        paths="data",
        crs=None,
        res=None,
        transforms=None,
        cache=True,
        time_series=False,
    ):
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load,
                here the collection of individual zip files for each tile
                should be found
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS (defaults to the
                resolution of the first file found)
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            time_series: if True, stack data along the time series dimension

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.paths = paths

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

    def _verify(self):
        """Verify the integrity of the dataset."""
        if self.files:
            return

        raise DatasetNotFoundError(self)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt
        import numpy as np
        from keras import ops

        mask = np.squeeze(ops.convert_to_numpy(sample["mask"]))
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = np.squeeze(ops.convert_to_numpy(sample["prediction"]))
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(prediction)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
