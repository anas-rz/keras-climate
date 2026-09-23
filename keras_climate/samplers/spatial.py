"""Spatial sampling routines (ported from torchgeo.samplers.spatial)."""

import math

import geopandas as gpd
import numpy as np
import shapely
from geopandas import GeoSeries

from .base import SpatialSampler
from .constants import Units
from .utils import _to_tuple, convolution_arithmetic


class RandomPatchSampler(SpatialSampler):
    """Sample locations from a region of interest randomly.

    This is particularly useful during training when you want to maximize
    the size of the dataset and return as many random chips as possible.
    Note that randomly sampled chips may overlap.
    """

    strategy = "random"

    def __init__(
        self, dataset, *, size, length=None, roi=None, units=Units.PIXELS, generator=None
    ):
        """Initialize a new RandomPatchSampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the
          height and width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is
          used for the height dimension, and the second *float* for the
          width dimension

        Args:
            dataset: Dataset to sample from.
            size: Dimensions of each patch.
            length: Number of random samples to draw per epoch (defaults to
                approximately the maximal number of non-overlapping chips of
                size ``size`` that could be sampled from the dataset).
            roi: Region of interest to sample from (defaults to the bounds
                of ``dataset.index``).
            units: Defines if ``size`` is in pixel or CRS units.
            generator: Pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, roi=roi)

        self.size = _to_tuple(size)
        self.generator = np.random.default_rng(generator)

        # Convert from pixel units to CRS units
        if units == Units.PIXELS:
            self.size = (self.size[0] * dataset.res[1], self.size[1] * dataset.res[0])

        # Default to approximate number of non-overlapping patches
        total_area = shapely.area(self.geometry)
        patch_area = self.size[0] * self.size[1]
        self._length = length or convolution_arithmetic(total_area, patch_area)

        # Erosion to avoid out-of-bounds sampling
        # Purposefully conservative radius calculation
        distance = math.sqrt((self.size[0] / 2) ** 2 + (self.size[1] / 2) ** 2)
        self.series = GeoSeries([shapely.buffer(self.geometry, -distance)])

    def __iter__(self):
        # Ensure a new set of random points for each epoch
        points = self.series.sample_points(size=len(self), rng=self.generator)

        # Points are random, but order is not
        points = points.explode().sample(frac=1, random_state=self.generator)

        # Snap to pixel grid
        xmin, ymin, _, _ = self.bounds
        # Convert from geospatial coords to pixel coords
        points = points.translate(-xmin - self.size[1] / 2, -ymin - self.size[0] / 2)
        points = points.scale(1 / self.res[0], 1 / self.res[1], origin=(0, 0))
        # Round to the nearest pixel
        x = points.x.round()
        y = points.y.round()
        # Convert from pixel coords to geospatial coords
        points = gpd.points_from_xy(x, y)
        points = points.scale(self.res[0], self.res[1], origin=(0, 0))
        points = points.translate(xmin, ymin)

        for point in points:
            xmin = point.x
            xmax = point.x + self.size[1]
            ymin = point.y
            ymax = point.y + self.size[0]
            yield slice(xmin, xmax), slice(ymin, ymax)


class GriddedPatchSampler(SpatialSampler):
    """Sample locations from a region of interest in a grid-like fashion.

    This is particularly useful during evaluation when you want to make
    predictions for an entire region of interest. You want to minimize the
    amount of redundant computation by minimizing overlap between chips.
    """

    strategy = "sequential"

    def __init__(self, dataset, *, size, stride=None, roi=None, units=Units.PIXELS):
        """Initialize a new GriddedPatchSampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the
          height and width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is
          used for the height dimension, and the second *float* for the
          width dimension

        Args:
            dataset: Dataset to sample from.
            size: Dimensions of each patch.
            stride: Distance to skip between each patch (defaults to
                *size*).
            roi: Region of interest to sample from (defaults to the bounds
                of ``dataset.index``).
            units: Defines if ``size`` and ``stride`` are in pixel or CRS
                units.
        """
        super().__init__(dataset, roi=roi)

        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride or size)

        # Convert from pixel units to CRS units
        if units == Units.PIXELS:
            self.size = (self.size[0] * dataset.res[1], self.size[1] * dataset.res[0])
            self.stride = (
                self.stride[0] * dataset.res[1],
                self.stride[1] * dataset.res[0],
            )

    def __iter__(self):
        xmin, ymin, xmax, ymax = self.geometry.bounds

        rows = convolution_arithmetic(ymax - ymin, self.size[0], self.stride[0])
        cols = convolution_arithmetic(xmax - xmin, self.size[1], self.stride[1])

        # Adjust xmin/ymin to have equal spacing in case of non-integer multiple
        geometry_width = xmax - xmin
        stride_width = self.stride[1] * (cols - 1) + self.size[1]
        xmin -= (stride_width - geometry_width) / 2
        geometry_height = ymax - ymin
        stride_height = self.stride[0] * (rows - 1) + self.size[0]
        ymin -= (stride_height - geometry_height) / 2

        # Snap to grid
        xmin = (xmin - self.bounds[0]) / self.res[0]
        ymin = (ymin - self.bounds[1]) / self.res[1]
        xmin = round(xmin)
        ymin = round(ymin)
        xmin = xmin * self.res[0] + self.bounds[0]
        ymin = ymin * self.res[1] + self.bounds[1]

        for i in range(rows):
            y = ymin + i * self.stride[0]

            for j in range(cols):
                x = xmin + j * self.stride[1]

                bbox = shapely.box(x, y, x + self.size[1], y + self.size[0])
                if self.geometry.intersects(bbox) and not self.geometry.touches(bbox):
                    yield slice(x, x + self.size[1]), slice(y, y + self.size[0])
