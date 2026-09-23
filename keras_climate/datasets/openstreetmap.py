"""OpenStreetMap dataset (ported from torchgeo.datasets.openstreetmap)."""

import contextlib
import hashlib
import json
import pathlib
import re
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from keras import ops
from matplotlib.colors import ListedColormap
from pyproj import CRS

from .errors import DatasetNotFoundError
from .geo import VectorDataset
from .utils import lazy_import


class OpenStreetMap(VectorDataset):
    """OpenStreetMap dataset.

    The `OpenStreetMap <https://www.openstreetmap.org/>`__ dataset provides
    access to crowd-sourced geographic data. This implementation uses the
    `Overpass API <https://wiki.openstreetmap.org/wiki/Overpass_API>`__ to
    query and download OSM data for a specified geographic bounding box at
    initialization, then allows spatial querying of the cached data.

    Dataset features:

    * Vector data (points, lines, polygons) for various geographic features
    * Flexible querying by class configuration (buildings, highways,
      amenities, etc.)
    * Data fetched once at initialization and cached locally
    * Class-based labeling with priority-based assignment

    Class priority and label assignment: ``classes`` is a list of dicts
    defining feature classes. Each has ``name`` (str) and ``selector`` (list
    of OSM tag filters). Features are assigned labels based on the order of
    classes in this list:

    - First class gets label=1, second gets label=2, etc.
    - If a feature matches multiple classes, it receives the label of the
      first matching class
    - Features that don't match any class get label=0 (background)

    Example::

        classes = [
            {"name": "buildings", "selector": [{"building": "*"}]},  # label=1
            {"name": "roads", "selector": [{"highway": "*"}]},  # label=2
        ]

    If you use this dataset in your research, please cite the following
    source:

    * https://www.openstreetmap.org/copyright
    """

    _overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    _min_request_interval = 1.0
    _last_request_time = 0.0

    def __init__(
        self,
        bbox,
        classes,
        paths="data",
        res=(0.0001, 0.0001),
        transforms=None,
        download=False,
    ):
        """Initialize a new OpenStreetMap dataset instance.

        Args:
            bbox: bounding box for initial data fetch as (xmin, ymin, xmax,
                ymax) in EPSG:4326
            classes: list of dicts defining feature classes. Each dict must
                have "name" (str) and "selector" (list of OSM tag filters).
                Features get labels 1-N based on class order, with first
                match taking priority.
            paths: paths directory where dataset will be stored
            res: resolution of the dataset in units of EPSG:4326 (degrees)
            transforms: a function/transform that takes input sample and
                returns a transformed version
            download: if True, download dataset and store it in the paths
                directory

        Raises:
            DatasetNotFoundError: if dataset is not found and download is
                False
            TypeError: if invalid class configuration
            ValueError: if invalid class configuration
        """
        self._validate_classes(classes)

        self.bbox = bbox
        self.classes = classes
        self.root = pathlib.Path(paths)
        self.root.mkdir(parents=True, exist_ok=True)

        if download:
            self._download_data()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        data_file = self._get_data_filename()

        super().__init__(
            paths=data_file,
            crs=CRS.from_epsg(4326),  # Always use WGS84 for OSM data
            res=res,
            transforms=transforms,
            label_name="label",
        )

        # Check for empty classes after initialization
        self._check_empty_classes()

    def _validate_classes(self, classes):
        """Validate classes configuration."""
        if not isinstance(classes, list) or not classes:
            raise TypeError("classes must be a non-empty list")

        for i, class_def in enumerate(classes):
            if not isinstance(class_def, dict):
                raise TypeError(f"Class {i} must be a dictionary")
            if "name" not in class_def or "selector" not in class_def:
                raise ValueError(f'Class {i} must have "name" and "selector" keys')
            if not isinstance(class_def["selector"], list):
                raise TypeError(f"Class {i} selector must be a list")
            for j, selector in enumerate(class_def["selector"]):
                if not isinstance(selector, dict):
                    raise TypeError(f"Class {i} selector {j} must be a dictionary")

    def _get_data_filename(self):
        """Get the filename for the cached data file."""
        cache_key = {"bbox": list(self.bbox), "classes": self.classes}
        cache_str = json.dumps(cache_key, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:16]
        return self.root / f"osm_features_{cache_hash}.geojson"

    def _check_integrity(self):
        """Check if the dataset file exists."""
        return self._get_data_filename().exists()

    def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = time.time()
        elapsed = current_time - OpenStreetMap._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        OpenStreetMap._last_request_time = time.time()

    def _build_overpass_query(self):
        """Build Overpass query from classes configuration."""
        xmin, ymin, xmax, ymax = self.bbox
        overpass_bbox = f"{ymin},{xmin},{ymax},{xmax}"

        queries = []
        for class_def in self.classes:
            for selector in class_def["selector"]:
                for tag, values in selector.items():
                    if values == "*":
                        queries.append(f'wr["{tag}"]({overpass_bbox});')
                    elif isinstance(values, list):
                        regex = f'^({"|".join(re.escape(v) for v in values)})$'
                        queries.append(f'wr["{tag}"~"{regex}"]({overpass_bbox});')
                    else:
                        queries.append(f'wr["{tag}"="{values}"]({overpass_bbox});')

        query = f"""
        [out:json][timeout:25];
        (
          {chr(10).join('  ' + q for q in queries)}
        );
        out tags geom;
        """

        return query.strip()

    def _download_data(self):
        """Download OSM data from Overpass API."""
        requests = lazy_import("requests")

        data_file = self._get_data_filename()

        if data_file.exists():
            return

        query = self._build_overpass_query()

        last_exception = None
        for endpoint in self._overpass_endpoints:
            try:
                self._rate_limit()

                headers = {
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "Accept": "application/json",
                }
                payload = {"data": query}

                response = requests.post(
                    endpoint, data=payload, headers=headers, timeout=30
                )
                response.raise_for_status()

                osm_data = response.json()
                gdf = self._parse_overpass_response(osm_data)

                if len(gdf) > 0:
                    gdf.to_file(data_file, driver="GeoJSON")
                    return
                else:
                    xmin, ymin, xmax, ymax = self.bbox
                    bbox_str = f"{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}"

                    msg = f"No features found in the specified area (bbox: {bbox_str}). "
                    msg += "Try a different feature type, larger bounding box, or different geographic area."
                    raise ValueError(msg)

            except ValueError:
                raise
            except Exception as e:  # noqa: BLE001
                last_exception = e
                continue

        raise RuntimeError(f"All Overpass API endpoints failed. Last error: {last_exception}")

    def _parse_overpass_response(self, osm_data):
        """Parse Overpass API response into a GeoDataFrame."""
        geometries = []
        properties = []

        for element in osm_data.get("elements", []):
            geom = self._element_to_geometry(element)
            if geom is not None:
                geometries.append(geom)
                props = element.get("tags", {}).copy()
                props["osm_id"] = element.get("id")
                props["osm_type"] = element.get("type")
                properties.append(props)

        if not geometries:
            return gpd.GeoDataFrame(
                columns=["geometry", "label"], geometry="geometry", crs="EPSG:4326"
            )

        labels = [self._get_class_label({"properties": props}) for props in properties]

        for props, label in zip(properties, labels):
            props["label"] = label

        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
        return gdf

    def _element_to_geometry(self, element):
        """Convert OSM element to Shapely geometry."""
        element_type = element.get("type")

        with contextlib.suppress(KeyError, ValueError, TypeError):
            if element_type == "node":
                lat = element.get("lat")
                lon = element.get("lon")
                if lat is not None and lon is not None:
                    return shapely.Point(lon, lat)

            elif element_type in ("way", "relation"):
                if "geometry" in element:
                    coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
                    if len(coords) >= 2:
                        tags = element.get("tags", {})
                        is_area = tags.get("area") == "yes" or (
                            len(coords) >= 4 and coords[0] == coords[-1]
                        )
                        return (
                            shapely.Polygon(coords)
                            if is_area
                            else shapely.LineString(coords)
                        )

        return None

    def _get_class_label(self, feature):
        """Get label based on class priority (first match wins)."""
        props = feature.get("properties", {})

        for class_idx, class_def in enumerate(self.classes):
            for selector in class_def["selector"]:
                if self._feature_matches_selector(props, selector):
                    return class_idx + 1

        return 0

    def _feature_matches_selector(self, props, selector):
        """Check if feature properties match a selector."""
        if "properties" in props and isinstance(props["properties"], str):
            try:
                actual_props = json.loads(props["properties"])
            except (json.JSONDecodeError, TypeError):
                actual_props = props
        else:
            actual_props = props

        for tag, expected_values in selector.items():
            if tag not in actual_props:
                return False

            actual_value = actual_props[tag]
            if actual_value is None:
                return False

            if pd.isna(actual_value):
                return False

            if expected_values == "*":
                continue
            elif isinstance(expected_values, list):
                if actual_value not in expected_values:
                    return False
            elif actual_value != expected_values:
                return False

        return True

    def _check_empty_classes(self):
        """Check for classes with no geometries and warn the user."""
        if self.classes:
            data_file = self._get_data_filename()
            gdf = gpd.read_file(data_file)

            if len(gdf) > 0:
                label_counts = gdf["label"].value_counts()

                for i, class_def in enumerate(self.classes):
                    class_label = i + 1
                    if label_counts.get(class_label, 0) == 0:
                        warnings.warn(
                            f"Class '{class_def['name']}' (label={class_label}) has no geometries in this AOI. "
                            "This may be due to no features of this type in the area or all features "
                            "being assigned to higher-priority classes.",
                            UserWarning,
                            stacklevel=3,
                        )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        mask = np.squeeze(ops.convert_to_numpy(sample["mask"]))
        ncols = 1

        showing_prediction = "prediction" in sample
        if showing_prediction:
            pred = np.squeeze(ops.convert_to_numpy(sample["prediction"]))
            ncols = 2

        colors = [
            "#000000",  # Background (label=0)
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FECA57",
            "#FF9FF3",
            "#54A0FF",
            "#5F27CD",
            "#00D2D3",
            "#FF3838",
            "#FF9500",
            "#7bed9f",
        ]

        cmap = ListedColormap(colors)

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 5))
        axs_list = axs if ncols > 1 else [axs]

        legend_handles = []
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:
                continue
            class_idx = int(label - 1)
            if class_idx < len(self.classes):
                class_name = self.classes[class_idx]["name"]
                color = colors[int(label) % len(colors)]
                legend_handles.append(mpatches.Patch(color=color, label=class_name.title()))

        axs_list[0].imshow(mask, cmap=cmap, vmin=0, vmax=len(colors) - 1, interpolation="none")
        axs_list[0].axis("off")
        if show_titles:
            axs_list[0].set_title("Mask")
        if legend_handles:
            axs_list[0].legend(handles=legend_handles, loc="upper right")

        if showing_prediction:
            axs_list[1].imshow(
                pred, cmap=cmap, vmin=0, vmax=len(colors) - 1, interpolation="none"
            )
            axs_list[1].axis("off")
            if show_titles:
                axs_list[1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
