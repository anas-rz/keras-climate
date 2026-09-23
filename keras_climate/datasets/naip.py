"""National Agriculture Imagery Program (NAIP) dataset (ported from
torchgeo.datasets.naip).
"""

from keras import ops

from .geo import RasterDataset


class NAIP(RasterDataset):
    """National Agriculture Imagery Program (NAIP) dataset.

    The `National Agriculture Imagery Program (NAIP)
    <https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip>`_
    acquires aerial imagery during the agricultural growing seasons in the
    continental U.S. A primary goal of the NAIP program is to make digital
    ortho photography available to governmental agencies and the public
    within a year of acquisition.

    NAIP is administered by the USDA's Farm Service Agency (FSA) through the
    Aerial Photography Field Office in Salt Lake City. This "leaf-on"
    imagery is used as a base layer for GIS programs in FSA's County Service
    Centers, and is used to maintain the Common Land Unit (CLU) boundaries.

    If you use this dataset in your research, please cite it using the
    following format:

    * https://www.fisheries.noaa.gov/inport/item/49508/citation
    """

    # https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs141p2_015644.pdf
    # https://planetarycomputer.microsoft.com/dataset/naip#Storage-Documentation
    filename_glob = "m_*.*"
    filename_regex = r"""
        ^m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>\d+)
        _(?P<date>\d+)
        (?:_(?P<processing_date>\d+))?
        \..*$
    """

    # Plotting
    all_bands = ("R", "G", "B", "NIR")
    rgb_bands = ("R", "G", "B")

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])[..., 0:3]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
