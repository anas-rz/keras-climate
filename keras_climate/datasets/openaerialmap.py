"""OpenAerialMap dataset (ported from torchgeo.datasets.openaerialmap)."""

import glob
import math
import os
import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from keras import ops
from pyproj import CRS as PROJ_CRS
from rasterio.crs import CRS as RIO_CRS
from rasterio.transform import from_bounds

from .geo import RasterDataset
from .utils import lazy_import


class TileUtils:
    """Web Mercator tile utilities for XYZ tile calculations.

    Implements standard Web Mercator (EPSG:3857) tile math for converting
    between geographic coordinates and tile indices.

    References:
        * OpenStreetMap Wiki - Slippy map tilenames:
          https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        * Web Mercator / Pseudo-Mercator (EPSG:3857):
          https://epsg.io/3857
        * OSGeo Tile Map Service Specification:
          https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification
    """

    Tile = namedtuple("Tile", ["x", "y", "z"])
    LngLatBbox = namedtuple("LngLatBbox", ["west", "south", "east", "north"])

    @classmethod
    def tile(cls, lng, lat, zoom, truncate=False):
        """Get tile coordinates containing a geographic point."""
        if truncate:
            lng = max(-180.0, min(180.0, lng))
            lat = max(-90.0, min(90.0, lat))

        x_frac = (lng + 180.0) / 360.0
        lat_rad = math.radians(lat)
        y_frac = (
            1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi
        ) / 2.0

        n = 2**zoom
        x = int(min(n - 1, max(0, math.floor(x_frac * n))))
        y = int(min(n - 1, max(0, math.floor(y_frac * n))))

        return cls.Tile(x, y, zoom)

    @classmethod
    def bounds(cls, t):
        """Get geographic bounds of a tile."""
        n = 2**t.z
        west = t.x / n * 360.0 - 180.0
        east = (t.x + 1) / n * 360.0 - 180.0

        north_rad = math.atan(math.sinh(math.pi * (1 - 2 * t.y / n)))
        south_rad = math.atan(math.sinh(math.pi * (1 - 2 * (t.y + 1) / n)))
        north = math.degrees(north_rad)
        south = math.degrees(south_rad)

        return cls.LngLatBbox(west, south, east, north)

    @classmethod
    def tiles(cls, west, south, east, north, zoom, truncate=False):
        """Generate tiles covering a bounding box."""
        if truncate:
            west = max(-180.0, min(180.0, west))
            south = max(-90.0, min(90.0, south))
            east = max(-180.0, min(180.0, east))
            north = max(-90.0, min(90.0, north))

        if west > east:
            bboxes = [(-180.0, south, east, north), (west, south, 180.0, north)]
        else:
            bboxes = [(west, south, east, north)]

        for w, s, e, n in bboxes:
            w = max(-180.0, w)
            s = max(-85.051129, s)
            e = min(180.0, e)
            n = min(85.051129, n)

            ul_tile = cls.tile(w, n, zoom)
            lr_tile = cls.tile(e, s, zoom)

            for x in range(ul_tile.x, lr_tile.x + 1):
                for y in range(ul_tile.y, lr_tile.y + 1):
                    yield cls.Tile(x, y, zoom)


class OpenAerialMap(RasterDataset):
    """OpenAerialMap dataset.

    The `OpenAerialMap (OAM) <https://openaerialmap.org/>`__ is an open
    service for accessing and sharing aerial imagery. The dataset provides
    access to crowd-sourced aerial imagery from various sources including
    drones, satellites, and aircraft.

    This implementation uses the `STAC API
    <https://api.imagery.hotosm.org/stac>`__ to query imagery and download
    tiles via TMS endpoints. The STAC API returns imagery sorted by most
    recent first.

    Dataset features:

    * Aerial imagery from various sources (drones, satellites, aircraft)
    * Global coverage with varying resolution
    * STAC-based querying (most recent imagery first)
    * Tile naming following mercantile standard: OAM-{x}-{y}-{z}.tif
    * Automatic georeferencing using rasterio
    * RGB imagery (3-band)

    Usage::

        dataset = OpenAerialMap(
            paths='data/openaerial',
            bbox=(lon_min, lat_min, lon_max, lat_max),
            zoom=19,
            download=True,
        )

    If you use this dataset in your research, please cite OpenAerialMap:

    * https://openaerialmap.org/
    """

    _stac_url = "https://api.imagery.hotosm.org/stac"
    _tiles_url = "https://api.imagery.hotosm.org/raster"

    filename_glob = "OAM-*.tif"

    all_bands = ("R", "G", "B")
    rgb_bands = ("R", "G", "B")

    def __init__(
        self,
        paths="data",
        crs=None,
        res=None,
        bbox=None,
        zoom=19,
        max_items=1,
        transforms=None,
        cache=True,
        download=False,
        image_id=None,
        tile_size=256,
    ):
        """Initialize a new OpenAerialMap dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to EPSG:4326)
            res: resolution of the dataset in units of CRS (defaults to
                resolution of first file found)
            bbox: bounding box for STAC query as (xmin, ymin, xmax, ymax) in
                EPSG:4326
            zoom: zoom level for tiles (15-23), only used when download=True
            max_items: maximum number of STAC items to query
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download imagery from STAC API based on bbox
            image_id: optional STAC item ID to download specific imagery
            tile_size: size of the tiles to download (256, 512, 768, 1024)

        Raises:
            DatasetNotFoundError: If dataset is not found and download=False.
            ValueError: If download=True but neither bbox nor image_id is
                provided.
        """
        self.paths = paths
        self.bbox = bbox
        self.zoom = zoom
        self.max_items = max_items
        self.download = download
        self.image_id = image_id
        self.tile_size = tile_size

        if download:
            if bbox is None:
                raise ValueError("bbox must be provided when download=True")
            if not 15 <= zoom <= 23:
                raise ValueError(f"zoom must be between 15 and 23, got {zoom}")
            if self._download():
                print("Download complete.")

        if crs is None:
            crs = PROJ_CRS.from_epsg(4326)

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def _download(self):
        """Download imagery from STAC API and TMS endpoints."""
        root = self.paths

        if isinstance(root, (str, os.PathLike)):
            os.makedirs(root, exist_ok=True)

        existing = glob.glob(os.path.join(root, self.filename_glob))
        if existing:
            print(f"Found {len(existing)} existing tiles, skipping download.")
            return False

        result = self._fetch_item_id()
        if not result:
            warnings.warn(
                f"No imagery found for bbox {self.bbox} or ID {self.image_id}. "
                "Try a different area or check OpenAerialMap coverage.",
                UserWarning,
                stacklevel=2,
            )
            return False

        west, south, east, north = self.bbox
        tiles = list(TileUtils.tiles(west, south, east, north, self.zoom, truncate=True))
        self._download_tiles(result, tiles)
        return True

    def _fetch_item_id(self):
        """Query STAC API and extract tiles URL."""
        requests = lazy_import("requests")

        params = {"limit": self.max_items}

        if self.image_id:
            params["ids"] = [self.image_id]
        elif self.bbox:
            params["bbox"] = list(self.bbox)

        try:
            response = requests.post(
                f"{self._stac_url}/search",
                json=params,
                headers={"User-Agent": "keras_climate"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to query STAC API: {e}") from e
        except (ValueError, KeyError) as e:
            raise RuntimeError(f"Invalid STAC API response: {e}") from e

        features = data.get("features", [])
        if not features:
            return None

        feature = features[0]
        props = feature.get("properties", {})
        item_id = feature.get("id")
        collection_id = feature.get("collection")

        if not item_id or not collection_id:
            return None

        print(f'Using OpenAerialMap image: {props.get("title", "Unknown")}')
        print(f"  ID: {item_id}")
        print(f"  Collection: {collection_id}")

        try:
            tiles_response = requests.get(
                f"{self._tiles_url}/collections/{collection_id}/items/{item_id}/tiles",
                headers={"User-Agent": "keras_climate"},
                timeout=30,
            )
            tiles_response.raise_for_status()
            tiles_data = tiles_response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to query tiles endpoint: {e}") from e

        tilesets = tiles_data.get("tilesets", [])
        for tileset in tilesets:
            links = tileset.get("links", [])
            for link in links:
                if link.get("rel") == "tile":
                    tile_url = link.get("href", "")
                    if "WebMercatorQuad" in tile_url:
                        return tile_url

        raise RuntimeError("WebMercatorQuad tileset not found in API response")

    def _download_tiles(self, tiles_url, tiles):
        """Download tiles concurrently using a thread pool."""
        print(f"Starting download of {len(tiles)} tiles...")
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._download_single_tile, tiles_url, tile): tile
                for tile in tiles
            }
            for future in as_completed(futures):
                future.result()
        print(f"Finished downloading {len(tiles)} tiles.")

    def _download_single_tile(self, tiles_url, tile):
        """Download and georeference a single tile."""
        requests = lazy_import("requests")

        root = self.paths

        url = (
            tiles_url.replace("{z}", str(tile.z))
            .replace("{x}", str(tile.x))
            .replace("{y}", str(tile.y))
        )
        url += f"@{int(self.tile_size / 256)}x"
        url += "?assets=visual"
        filename = f"OAM-{tile.x}-{tile.y}-{tile.z}.tif"
        filepath = os.path.join(root, filename)

        try:
            response = requests.get(url, headers={"User-Agent": "keras_climate"}, timeout=30)
            if response.status_code != 200:
                warnings.warn(
                    f"Failed to download tile {tile}: HTTP {response.status_code}",
                    UserWarning,
                )
                return

            with open(filepath, "wb") as f:
                f.write(response.content)

            self._georeference_tile(filepath, tile)

        except (requests.RequestException, OSError) as e:
            warnings.warn(f"Error downloading tile {tile}: {e}", UserWarning)

    def _georeference_tile(self, filepath, tile):
        """Add georeferencing metadata to a downloaded tile."""
        import rasterio

        bounds = TileUtils.bounds(tile)
        try:
            with rasterio.open(filepath, "r+") as dataset:
                dataset.transform = from_bounds(
                    bounds.west,
                    bounds.south,
                    bounds.east,
                    bounds.north,
                    dataset.width,
                    dataset.height,
                )
                dataset.crs = RIO_CRS.from_epsg(4326)
                dataset.update_tags(
                    ns="rio_georeference",
                    georeferencing_applied="True",
                    tile_x=str(tile.x),
                    tile_y=str(tile.y),
                    tile_z=str(tile.z),
                )
        except rasterio.errors.RasterioIOError:
            warnings.warn(
                f"Could not georeference {filepath}. Not a valid raster file.",
                UserWarning,
                stacklevel=2,
            )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt
        import numpy as np

        rgb = ops.convert_to_numpy(sample["image"])[..., 0:3]

        if np.issubdtype(rgb.dtype, np.floating) and rgb.max() > 1:
            rgb = np.clip(rgb / 255.0, 0, 1)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        ax.imshow(rgb)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
