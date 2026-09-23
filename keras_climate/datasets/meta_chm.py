"""Meta Canopy Height Map (CHM) v2 dataset (ported from torchgeo.datasets.meta_chm)."""

import io
import urllib.request

import geopandas as gpd
import pandas as pd
from keras import ops

from .geo import RasterDataset


class MetaCHM(RasterDataset):
    """Meta Canopy Height Map (CHM) v2 dataset.

    The `Meta CHMv2 (DINOv3) global canopy height map
    <https://ai.meta.com/ai-for-good/datasets/canopy-height-maps/>`__ is a
    global, ~1.19 m/pixel estimate of tree canopy height derived from
    high-resolution satellite imagery.

    Dataset features:

    * canopy height in meters at ~1.19 m/pixel (uint8, 0 = no canopy or no
      data)
    * 213,109 tiles on a zoom-10 Web Mercator (EPSG:3857) grid

    Dataset format:

    * a STAC GeoParquet index with each tile's geometry, acquisition date,
      and COG URL
    * single-channel uint8 Cloud-Optimized GeoTIFFs whose pixel values are
      the canopy height in meters (0 = no canopy or no data)

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2603.06382
    """

    url = "https://data.source.coop/tge-labs/meta-chm-v2/stac/items.parquet"

    is_image = False
    # Overrides RasterDataset.dtype (a property) with a plain attribute, since
    # this is a float-valued mask (canopy height in meters), not an integer one.
    dtype = "float32"
    all_bands = ("chm",)
    _res = (1.1943285669558463, 1.1943285669558463)

    def __init__(self, transforms=None, cache=True):
        """Initialize a new MetaCHM instance.

        Args:
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
        """
        self.paths = self.url
        self.transforms = transforms
        self.cache = cache
        self.bands = self.all_bands
        self.band_indexes = None
        self.time_series = False

        request = urllib.request.Request(self.url, headers={"User-Agent": "keras_climate"})
        with urllib.request.urlopen(request) as response:
            buffer = io.BytesIO(response.read())
        columns = ["geometry", "assets", "datetime"]
        gdf = gpd.read_parquet(buffer, columns=columns)
        gdf.to_crs("EPSG:3857", inplace=True)
        filepaths = (
            gdf["assets"]
            .map(lambda asset: asset["chm"]["href"])
            .str.replace(
                "s3://dataforgood-fb-data/",
                "https://dataforgood-fb-data.s3.amazonaws.com/",
                regex=False,
            )
        )
        datetimes = gdf["datetime"].dt.tz_localize(None)
        index = pd.IntervalIndex.from_arrays(
            datetimes, datetimes + pd.Timedelta(days=1), closed="both", name="datetime"
        )
        self.index = gpd.GeoDataFrame(
            {"filepath": filepaths.to_numpy()},
            index=index,
            geometry=gdf.geometry.to_numpy(),
            crs=gdf.crs,
        )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        mask = ops.convert_to_numpy(sample["mask"])
        ncols = 1

        showing_prediction = "prediction" in sample
        if showing_prediction:
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs = [axs] if ncols == 1 else axs

        im = axs[0].imshow(mask, cmap="YlGn", vmin=0, vmax=40)
        axs[0].axis("off")
        fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, label="Canopy height (m)")
        if show_titles:
            axs[0].set_title("Canopy Height")

        if showing_prediction:
            pred = ops.convert_to_numpy(sample["prediction"])
            im = axs[1].imshow(pred, cmap="YlGn", vmin=0, vmax=40)
            axs[1].axis("off")
            fig.colorbar(
                im, ax=axs[1], fraction=0.046, pad=0.04, label="Canopy height (m)"
            )
            if show_titles:
                axs[1].set_title("Prediction")

        if suptitle is not None:
            fig.suptitle(suptitle)
        return fig
