"""Chesapeake Bay Program Land Use/Land Cover Data Project datasets.

(ported from torchgeo.datasets.chesapeake)
"""

import abc
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.windows
import shapely.geometry
import shapely.ops
from keras import ops
from matplotlib.colors import ListedColormap

from .errors import DatasetNotFoundError
from .geo import GeoDataset, RasterDataset
from .utils import download_url, extract_archive


class Chesapeake(RasterDataset, abc.ABC):
    """Abstract base class for all Chesapeake datasets.

    `Chesapeake Bay Land Use and Land Cover (LULC) Database 2022 Edition
    <https://www.chesapeakeconservancy.org/projects/cbp-land-use-land-cover-data-project>`_

    The Chesapeake Bay Land Use and Land Cover Database (LULC) facilitates
    characterization of the landscape and land change for and between
    discrete time periods. The database contains one-meter 13-class Land
    Cover (LC) and 54-class Land Use/Land Cover (LULC) for all counties
    within or adjacent to the Chesapeake Bay watershed for 2013/14 and
    2017/18, depending on availability of National Agricultural Imagery
    Program (NAIP) imagery for each state.

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.5066/P981GV1L
    """

    url = "https://hf.co/datasets/torchgeo/chesapeake/resolve/1e0370eda6a24d93af4153745e54fd383d015bf5/{state}_lulc_{year}_2022-Edition.zip"
    filename_glob = "{state}_lulc_*_2022-Edition.tif"
    filename_regex = r"^{state}_lulc_(?P<date>\d{{4}})_2022-Edition\.tif$"
    date_format = "%Y"
    is_image = False

    @property
    @abc.abstractmethod
    def sha256s(self):
        """Mapping between data year and zip file sha256."""

    @property
    def state(self):
        """State abbreviation."""
        return self.__class__.__name__[-2:].lower()

    cmap = ListedColormap(
        np.array(
            [
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (0, 92, 230, 255),
                (0, 92, 230, 255),
                (0, 92, 230, 255),
                (0, 92, 230, 255),
                (0, 92, 230, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (0, 0, 0, 255),
                (235, 6, 2, 255),
                (89, 89, 89, 255),
                (138, 138, 136, 255),
                (138, 138, 136, 255),
                (138, 138, 136, 255),
                (115, 115, 0, 255),
                (233, 255, 190, 255),
                (255, 255, 115, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (38, 115, 0, 255),
                (56, 168, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 115, 255),
                (255, 255, 115, 255),
                (255, 255, 115, 255),
                (170, 255, 0, 255),
                (170, 255, 0, 255),
                (170, 255, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (77, 209, 148, 255),
                (77, 209, 148, 255),
                (56, 168, 0, 255),
                (38, 115, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (186, 245, 217, 255),
                (186, 245, 217, 255),
                (56, 168, 0, 255),
                (38, 115, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 211, 127, 255),
                (255, 211, 127, 255),
                (255, 211, 127, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (0, 168, 132, 255),
                (0, 168, 132, 255),
                (0, 168, 132, 255),
                (56, 168, 0, 255),
                (38, 115, 0, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
                (255, 255, 255, 255),
            ]
        )
        / 255
    )

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
        """Initialize a new Chesapeake instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres)
                format (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)
            time_series: if True, stack data along the time series dimension
                [T, H, W, C]. If False, merge data into a [H, W, C] mosaic.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.filename_glob = self.filename_glob.format(state=self.state)
        self.filename_regex = self.filename_regex.format(state=self.state)

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

        paths = self.paths
        if glob.glob(os.path.join(paths, "**", "*.zip"), recursive=True):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self):
        """Download the dataset."""
        paths = self.paths
        for year, sha256 in self.sha256s.items():
            url = self.url.format(state=self.state, year=year)
            download_url(url, paths, sha256=sha256 if self.checksum else None)

    def _extract(self):
        """Extract the dataset."""
        paths = self.paths
        for file in glob.iglob(os.path.join(paths, "**", "*.zip"), recursive=True):
            extract_archive(file)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        mask = ops.convert_to_numpy(sample["mask"])
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = ops.convert_to_numpy(sample["prediction"])
            ncols = 2

        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(4 * ncols, 4))
        kwargs = {"cmap": self.cmap, "vmin": 0, "vmax": 128, "interpolation": "none"}

        axs[0, 0].imshow(mask, **kwargs)
        axs[0, 0].axis("off")
        if show_titles:
            axs[0, 0].set_title("Mask")

        if showing_predictions:
            axs[0, 1].imshow(pred, **kwargs)
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class ChesapeakeDC(Chesapeake):
    """This subset of the dataset contains data only for Washington, D.C."""

    sha256s = {
        2013: "13c776f8690a19c4088df6f4c143bca9650ce6cfe5c4091f3b774441eab8805a",
        2017: "f71ae8836bbbf7e4b60e4edf99e3a6eee16f62473dcb2c7f20ba5836c7a108c8",
    }


class ChesapeakeDE(Chesapeake):
    """This subset of the dataset contains data only for Delaware."""

    sha256s = {
        2013: "ced3e274bfd8531915cb21d1a3faad31c9de859648feb8b8daec245f343c0b5c",
        2018: "4b996051cbd532dc4e43642d4ecbdf2dd55456a927edc35983b427f137145273",
    }


class ChesapeakeMD(Chesapeake):
    """This subset of the dataset contains data only for Maryland."""

    sha256s = {
        2013: "e5a6a2c02f50295f6f13539092ce801c94263bd7ac304203cc2da2d4d774dd18",
        2018: "d3b70ae09737e119f5292b071f752add4c44c1f0a0b959a26305f2ceb0c583ec",
    }


class ChesapeakeNY(Chesapeake):
    """This subset of the dataset contains data only for New York."""

    sha256s = {
        2013: "2b861e5893f9340fc792c3df2dedfd0230ee35b1cd22fb53cc258fa7a97f4d33",
        2017: "eceaae776c49636730a51b5abe93ec123000535e92aee37cf9594c8ba52cd41e",
    }


class ChesapeakePA(Chesapeake):
    """This subset of the dataset contains data only for Pennsylvania."""

    sha256s = {
        2013: "5f197d18765ff8e0c837433c2eba285f97a1c5ac68c12d8c06fc4854c5df3d8c",
        2017: "f699ea705ae5bcda04e0eaa5c14369bccc1cec7ad65b1c4946134e8c55b84737",
    }


class ChesapeakeVA(Chesapeake):
    """This subset of the dataset contains data only for Virginia."""

    sha256s = {
        2014: "9e71799f70c4c994eadeb126e4923e7b843dff335b46cb27a4785f1551660226",
        2018: "c7dd3514268f83fe8a3d650b3e686f4a7bce091b04f6e16a84e4f3d475bf19e9",
    }


class ChesapeakeWV(Chesapeake):
    """This subset of the dataset contains data only for West Virginia."""

    sha256s = {
        2014: "a882edc5e71acae0da953e44266562a0a957c8b139a885a20bbd33b22fe43d42",
        2018: "84c3577680bba0c2da179b0def119f014581cdf08d16039188d6e482a997cb8d",
    }


class ChesapeakeCVPR(GeoDataset):
    """CVPR 2019 Chesapeake Land Cover dataset.

    The `CVPR 2019 Chesapeake Land Cover
    <https://lila.science/datasets/chesapeakelandcover>`_ dataset contains two layers of
    NAIP aerial imagery, Landsat 8 leaf-on and leaf-off imagery, Chesapeake Bay land
    cover labels, NLCD land cover labels, and Microsoft building footprint labels.

    This dataset was organized to accompany the 2019 CVPR paper, "Large Scale
    High-Resolution Land Cover Mapping with Multi-Resolution Data".

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/cvpr.2019.01301
    """

    subdatasets = ("base", "prior_extension")
    urls = {
        "base": "https://lilawildlife.blob.core.windows.net/lila-wildlife/lcmcvpr2019/cvpr_chesapeake_landcover.zip",
        "prior_extension": "https://zenodo.org/records/5866525/files/cvpr_chesapeake_landcover_prior_extension.zip?download=1",
    }
    filenames = {
        "base": "cvpr_chesapeake_landcover.zip",
        "prior_extension": "cvpr_chesapeake_landcover_prior_extension.zip",
    }
    md5s = {"base": "1225ccbb9590e9396875f221e5031514", "prior_extension": "402f41d07823c8faf7ea6960d7c4e17a"}

    _res = (1, 1)

    lc_cmap = ListedColormap(
        np.array(
            [
                (0, 0, 0, 0),
                (0, 197, 255, 255),
                (38, 115, 0, 255),
                (163, 255, 115, 255),
                (255, 170, 0, 255),
                (156, 156, 156, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            ]
        )
        / 255
    )

    prior_color_matrix = np.array(
        [
            [0.0, 0.77254902, 1.0, 1.0],
            [0.14901961, 0.45098039, 0.0, 1.0],
            [0.63921569, 1.0, 0.45098039, 1.0],
            [0.61176471, 0.61176471, 0.61176471, 1.0],
        ]
    )

    valid_layers = (
        "naip-new",
        "naip-old",
        "landsat-leaf-on",
        "landsat-leaf-off",
        "nlcd",
        "lc",
        "buildings",
        "prior_from_cooccurrences_101_31_no_osm_no_buildings",
    )
    states = ("de", "md", "va", "wv", "pa", "ny")
    splits = (
        [f"{state}-train" for state in states]
        + [f"{state}-val" for state in states]
        + [f"{state}-test" for state in states]
    )

    # the layer that is only distributed in the prior extension archive
    prior_layer = "prior_from_cooccurrences_101_31_no_osm_no_buildings"

    # these are used to check the integrity of each subdataset
    _files = {
        "base": (
            "de_1m_2013_extended-debuffered-test_tiles",
            "de_1m_2013_extended-debuffered-train_tiles",
            "de_1m_2013_extended-debuffered-val_tiles",
            "md_1m_2013_extended-debuffered-test_tiles",
            "md_1m_2013_extended-debuffered-train_tiles",
            "md_1m_2013_extended-debuffered-val_tiles",
            "ny_1m_2013_extended-debuffered-test_tiles",
            "ny_1m_2013_extended-debuffered-train_tiles",
            "ny_1m_2013_extended-debuffered-val_tiles",
            "pa_1m_2013_extended-debuffered-test_tiles",
            "pa_1m_2013_extended-debuffered-train_tiles",
            "pa_1m_2013_extended-debuffered-val_tiles",
            "va_1m_2014_extended-debuffered-test_tiles",
            "va_1m_2014_extended-debuffered-train_tiles",
            "va_1m_2014_extended-debuffered-val_tiles",
            "wv_1m_2014_extended-debuffered-test_tiles",
            "wv_1m_2014_extended-debuffered-train_tiles",
            "wv_1m_2014_extended-debuffered-val_tiles",
            "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_buildings.tif",
            "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-off.tif",
            "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-on.tif",
            "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_lc.tif",
            "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-new.tif",
            "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-old.tif",
            "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_nlcd.tif",
            "spatial_index.geojson",
        ),
        "prior_extension": (
            "wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif",
        ),
    }

    p_src_crs = pyproj.CRS("epsg:3857")
    p_transformers = {
        "epsg:26917": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26917"), always_xy=True
        ),
        "epsg:26918": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26918"), always_xy=True
        ),
    }

    def __init__(
        self,
        root="data",
        splits=("de-train",),
        layers=("naip-new", "lc"),
        transforms=None,
        cache=True,
        download=False,
        checksum=True,
    ):
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            splits: a list of strings in the format "{state}-{train,val,test}"
                indicating the subset of data to use, for example "ny-train"
            layers: a list containing a subset of "naip-new", "naip-old",
                "lc", "nlcd", "landsat-leaf-on", "landsat-leaf-off",
                "buildings", or
                "prior_from_cooccurrences_101_31_no_osm_no_buildings"
                indicating which layers to load
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: if ``splits`` or ``layers`` are not valid
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        for split in splits:
            assert split in self.splits
        assert all(layer in self.valid_layers for layer in layers)
        self.root = root
        self.layers = layers
        self.transforms = transforms
        self.cache = cache
        self.download = download
        self.checksum = checksum

        if self.prior_layer in layers:
            self.subdatasets = ("base", "prior_extension")
        else:
            self.subdatasets = ("base",)

        self._verify()

        # Add all tiles into the index in epsg:3857 based on the included geojson
        mint = pd.Timestamp.min
        maxt = pd.Timestamp.max
        gdf = gpd.read_file(os.path.join(root, "spatial_index.geojson"))
        gdf = gdf[gdf["split"].isin(splits)]
        gdf["prior_from_cooccurrences_101_31_no_osm_no_buildings"] = gdf["lc"].str.replace(
            "lc.tif", "prior_from_cooccurrences_101_31_no_osm_no_buildings.tif"
        )
        datetimes = [(mint, maxt)] * len(gdf)
        index = pd.IntervalIndex.from_tuples(datetimes, closed="both", name="datetime")
        gdf.set_crs("EPSG:3857", inplace=True)
        gdf.set_index(index, inplace=True)
        self.index = gdf

    def __getitem__(self, index):
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice."""
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            "bounds": self._slice_to_tensor(index),
            "transform": ops.convert_to_tensor(np.array(list(transform))),
        }

        images = []
        masks = []
        if df.empty:
            raise IndexError(
                f"index: {index} not found in dataset with bounds: {self.bounds}"
            )
        elif len(df) == 1:
            filenames = df.iloc[0]
            query_box_transformed = None  # is set by the first layer

            query_box = shapely.geometry.box(x.start, y.start, x.stop, y.stop)

            for layer in self.layers:
                fn = filenames[layer]

                with rasterio.open(os.path.join(self.root, fn)) as f:
                    dst_crs = f.crs.to_string().lower()

                    if query_box_transformed is None:
                        query_box_transformed = shapely.ops.transform(
                            self.p_transformers[dst_crs].transform, query_box
                        ).envelope

                    # Use a boundless windowed read so the returned array
                    # always matches the requested patch shape, even when
                    # the query extends beyond the raster footprint. The
                    # out-of-raster region is filled with nodata (or 0 if
                    # the raster has no nodata defined).
                    window = rasterio.windows.from_bounds(
                        *query_box_transformed.bounds, transform=f.transform
                    )
                    out_height = round(window.height)
                    out_width = round(window.width)
                    fill_value = f.nodata if f.nodata is not None else 0
                    data = f.read(
                        window=window,
                        boundless=True,
                        fill_value=fill_value,
                        out_shape=(f.count, out_height, out_width),
                    )
                    # (C, H, W) -> (H, W, C)
                    data = np.transpose(data, (1, 2, 0))

                if layer in ["naip-new", "naip-old", "landsat-leaf-on", "landsat-leaf-off"]:
                    images.append(data)
                elif layer in [
                    "lc",
                    "nlcd",
                    "buildings",
                    "prior_from_cooccurrences_101_31_no_osm_no_buildings",
                ]:
                    masks.append(data)
        else:
            raise IndexError(f"index: {index} spans multiple tiles which is not valid")

        image = np.concatenate(images, axis=-1).astype("float32")
        mask = np.concatenate(masks, axis=-1).astype("int64")
        if mask.shape[-1] == 1:
            mask = mask[..., 0]

        sample["image"] = ops.convert_to_tensor(image)
        sample["mask"] = ops.convert_to_tensor(mask)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""

        def exists(filename):
            return os.path.exists(os.path.join(self.root, filename))

        if all(
            exists(filename)
            for subdataset in self.subdatasets
            for filename in self._files[subdataset]
        ):
            return

        if all(
            os.path.exists(os.path.join(self.root, self.filenames[subdataset]))
            for subdataset in self.subdatasets
        ):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self):
        """Download the dataset."""
        for subdataset in self.subdatasets:
            download_url(
                self.urls[subdataset],
                self.root,
                filename=self.filenames[subdataset],
                md5=self.md5s[subdataset],
            )

    def _extract(self):
        """Extract the dataset."""
        for subdataset in self.subdatasets:
            extract_archive(os.path.join(self.root, self.filenames[subdataset]))

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        mask = ops.convert_to_numpy(sample["mask"])
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]

        num_panels = len(self.layers)
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = ops.convert_to_numpy(sample["prediction"])
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))

        i = 0
        for layer in self.layers:
            if layer == "naip-new" or layer == "naip-old":
                img = image[:, :, :3] / 255
                image = image[:, :, 4:]
                axs[i].axis("off")
                axs[i].imshow(img)
            elif layer == "landsat-leaf-on" or layer == "landsat-leaf-off":
                img = image[:, :, [3, 2, 1]] / 3000
                image = image[:, :, 9:]
                axs[i].axis("off")
                axs[i].imshow(img)
            elif layer == "nlcd":
                from .nlcd import NLCD

                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[i].imshow(img, vmin=0, vmax=255, cmap=NLCD.cmap, interpolation="none")
                axs[i].axis("off")
            elif layer == "lc":
                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[i].imshow(
                    img, vmin=0, vmax=15, cmap=self.lc_cmap, interpolation="none"
                )
                axs[i].axis("off")
            elif layer == "buildings":
                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[i].imshow(img, vmin=0, vmax=1, cmap="gray", interpolation="none")
                axs[i].axis("off")
            elif layer == "prior_from_cooccurrences_101_31_no_osm_no_buildings":
                img = (mask[:, :, :4] @ self.prior_color_matrix) / 255
                mask = mask[:, :, 4:]
                axs[i].imshow(img)
                axs[i].axis("off")

            if show_titles:
                if layer == "prior_from_cooccurrences_101_31_no_osm_no_buildings":
                    axs[i].set_title("prior")
                else:
                    axs[i].set_title(layer)
            i += 1

        if showing_predictions:
            axs[i].imshow(predictions, vmin=0, vmax=15, cmap=self.lc_cmap, interpolation="none")
            axs[i].axis("off")
            if show_titles:
                axs[i].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
