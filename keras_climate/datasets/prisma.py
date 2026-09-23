"""PRISMA dataset (ported from torchgeo.datasets.prisma)."""

from .geo import RasterDataset
from .utils import quantile_normalization


class PRISMA(RasterDataset):
    """PRISMA dataset.

    Hyperspectral Precursor and Application Mission
    `PRISMA <https://www.eoportal.org/satellite-missions/prisma-hyperspectral>`__
    (PRecursore IperSpettrale della Missione Applicativa) is a
    medium-resolution hyperspectral imaging satellite, developed, owned, and
    operated by the Italian Space Agency ASI.

    PRISMA carries two sensor instruments, the HYC (Hyperspectral Camera)
    module and the PAN (Panchromatic Camera) module. The HYC sensor is a
    prism spectrometer for two bands, VIS/NIR and NIR/SWIR, with a total of
    237 channels across both bands, at 30 m spatial resolution.

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1109/IGARSS.2018.8517785

    .. note::
       PRISMA imagery is distributed as HDF5 files. This loader requires you
       to first convert all files from HDF5 to GeoTIFF using something like
       `this script
       <https://gist.github.com/adamjstewart/7f5324a6a339c20e778f39536402fb4a>`__.
    """

    # https://prisma.asi.it/missionselect/docs/PRISMA%20Product%20Specifications_Is2_3.pdf
    #
    # See sections:
    #
    # * 6.3.6: L0A Product Naming Convention
    # * 7.5:   L1 Product Naming Convention
    # * 7.8.5: FKDP, GKDP, ICU-KDP and CDP Products Naming Convention
    filename_glob = "PRS_*"
    filename_regex = r"""
        ^PRS
        _(?P<level>[A-Z\d]+)
        _(?P<product>[A-Z]+)
        (_(?P<order>[A-Z_]+))?
        _(?P<start>\d{14})
        _(?P<stop>\d{14})
        _(?P<version>\d{4})
        (_(?P<valid>\d))?
        \.
    """
    date_format = "%Y%m%d%H%M%S"

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt
        from keras import ops

        # RGB band indices based on https://doi.org/10.3390/rs14164080
        rgb_indices = [34, 23, 11]
        image = ops.take(sample["image"], rgb_indices, axis=-1)
        image = ops.cast(image, "float32")
        image = quantile_normalization(image)
        image = ops.convert_to_numpy(image)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
