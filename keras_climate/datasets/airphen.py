"""Airphen dataset (ported from torchgeo.datasets.airphen)."""

from keras import ops

from .errors import RGBBandsMissingError
from .geo import RasterDataset
from .utils import quantile_normalization


class Airphen(RasterDataset):
    """Airphen dataset.

    `Airphen <https://ideol.sakura.ne.jp/img/20170123_HiphenAirphenKeyfeatures.pdf>`__
    is a multispectral scientific camera developed by agronomists and
    photonics engineers at `Hiphen <https://www.hiphen-plant.com/>`_ to match
    plant measurement needs and constraints.

    Main characteristics:

    * 6 Synchronized global shutter sensors
    * Sensor resolution 1280 x 960 pixels
    * Data format (.tiff, 12 bit)
    * SD card storage
    * Metadata information: Exif and XMP
    * Internal or external GPS
    * Synchronization with different sensors (TIR, RGB, others)

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.34133/2021/9892647
    """

    # Each camera measures a custom set of spectral bands chosen at purchase
    # time. Hiphen offers 8 bands to choose from, sorted from short to long
    # wavelength.
    all_bands = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
    rgb_bands = ("B4", "B3", "B1")

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        import matplotlib.pyplot as plt

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = ops.take(sample["image"], rgb_indices, axis=-1)
        image = ops.cast(image, "float32")
        image = quantile_normalization(image)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(ops.convert_to_numpy(image))
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
