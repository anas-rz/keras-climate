"""Deprecated single-sample geo samplers (ported from torchgeo.samplers.single)."""

import abc

import numpy as np
import pandas as pd
import shapely

from .constants import Units
from .utils import _to_tuple, get_random_bounding_box, tile_to_chips


class GeoSampler(abc.ABC):
    """Abstract base class for sampling from a :class:`~keras_climate.datasets.GeoDataset`.

    .. deprecated:: Use :class:`~keras_climate.samplers.base.GeoSampler` (via
       :class:`~keras_climate.samplers.spatial.RandomPatchSampler` /
       :class:`~keras_climate.samplers.spatial.GriddedPatchSampler`) instead.
    """

    def __init__(self, dataset, roi=None, toi=None):
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (defaults to the bounds
                of ``dataset.index``)
            toi: time of interest to sample from (defaults to the bounds of
                ``dataset.index``)
        """
        self.index = dataset.index
        self.res = dataset.res

        if roi:
            self.roi = roi
            self.index = self.index.clip(roi)
        else:
            x, y, t = dataset.bounds
            self.roi = shapely.box(x.start, y.start, x.stop, y.stop)

        if toi:
            self.toi = toi
            self.index = self.index.iloc[self.index.index.overlaps(toi)]
            tmin = np.maximum(self.index.index.left, np.datetime64(toi.left))
            tmax = np.minimum(self.index.index.right, np.datetime64(toi.right))
            self.index.index = pd.IntervalIndex.from_arrays(
                tmin, tmax, closed="both", name="datetime"
            )
        else:
            x, y, t = dataset.bounds
            self.toi = pd.Interval(t.start, t.stop)

    @abc.abstractmethod
    def __iter__(self):
        """Return the index of a dataset."""


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    .. deprecated:: Use :class:`~keras_climate.samplers.RandomPatchSampler` instead.

    This is particularly useful during training when you want to maximize
    the size of the dataset and return as many random chips as possible.
    Note that randomly sampled chips may overlap. This sampler is not
    recommended for use with tile-based datasets; use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset,
        size,
        length=None,
        roi=None,
        toi=None,
        units=Units.PIXELS,
        generator=None,
    ):
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the
          height and width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is
          used for the height dimension, and the second *float* for the
          width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each patch
            length: number of random samples to draw per epoch (defaults to
                approximately the maximal number of non-overlapping chips of
                size ``size`` that could be sampled from the dataset)
            roi: region of interest to sample from (defaults to the bounds
                of ``dataset.index``)
            toi: time of interest to sample from (defaults to the bounds of
                ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            generator: pseudo-random number generator (PRNG)
        """
        super().__init__(dataset, roi, toi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])

        self.generator = np.random.default_rng(generator)
        self.length = 0
        self.bounds = []
        self.intervals = []
        areas = []
        for hit in range(len(self.index)):
            bounds = self.index.geometry.iloc[hit].bounds
            xmin, ymin, xmax, ymax = bounds
            tmin, tmax = self.index.index[hit].left, self.index.index[hit].right
            if xmax - xmin >= self.size[1] and ymax - ymin >= self.size[0]:
                if xmax > xmin and ymax > ymin:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.bounds.append(bounds)
                self.intervals.append(pd.Interval(tmin, tmax))
                areas.append((xmax - xmin) * (ymax - ymin))

        if length is not None:
            self.length = length

        # Sampling requires float probabilities > 0
        self.areas = np.array(areas, dtype="float64")
        if self.areas.sum() == 0:
            self.areas += 1

    def __iter__(self):
        probs = self.areas / self.areas.sum()
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = self.generator.choice(len(self.areas), p=probs)
            bounds = self.bounds[idx]
            interval = self.intervals[idx]

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(
                bounds, self.size, self.res, self.generator
            )

            yield *bounding_box, slice(interval.left, interval.right)

    def __len__(self):
        return self.length


class GridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    .. deprecated:: Use :class:`~keras_climate.samplers.GriddedPatchSampler` instead.

    This is particularly useful during evaluation when you want to make
    predictions for an entire region of interest. You want to minimize the
    amount of redundant computation by minimizing overlap between chips.
    Usually the stride should be slightly smaller than the chip size such
    that each chip has some small overlap with surrounding chips.
    """

    def __init__(
        self, dataset, size, stride=None, roi=None, toi=None, units=Units.PIXELS
    ):
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the
          height and width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is
          used for the height dimension, and the second *float* for the
          width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each patch
            stride: distance to skip between each patch (defaults to *size*)
            roi: region of interest to sample from (defaults to the bounds
                of ``dataset.index``)
            toi: time of interest to sample from (defaults to the bounds of
                ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi, toi)
        self.size = _to_tuple(size)
        if stride is not None:
            self.stride = _to_tuple(stride)
        else:
            self.stride = self.size

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])
            self.stride = (self.stride[0] * self.res[1], self.stride[1] * self.res[0])

        self.length = 0
        for i in range(len(self.index)):
            bounds = self.index.geometry.iloc[i].bounds
            xmin, ymin, xmax, ymax = bounds
            if xmax - xmin < self.size[1] or ymax - ymin < self.size[0]:
                continue
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols

    def __iter__(self):
        # For each tile...
        for i in range(len(self.index)):
            bounds = self.index.geometry.iloc[i].bounds
            xmin, ymin, xmax, ymax = bounds
            if xmax - xmin < self.size[1] or ymax - ymin < self.size[0]:
                continue
            tmin, tmax = self.index.index[i].left, self.index.index[i].right
            rows, cols = tile_to_chips(bounds, self.size, self.stride)

            for i in range(rows):
                ymin = bounds[1] + i * self.stride[0]
                ymax = ymin + self.size[0]

                for j in range(cols):
                    xmin = bounds[0] + j * self.stride[1]
                    xmax = xmin + self.size[1]

                    yield slice(xmin, xmax), slice(ymin, ymax), slice(tmin, tmax)

    def __len__(self):
        return self.length


class PreChippedGeoSampler(GeoSampler):
    """Samples entire files at a time.

    .. deprecated:: Use :class:`~keras_climate.samplers.RandomPatchSampler` instead.

    This is particularly useful for datasets that contain geospatial
    metadata and subclass :class:`~keras_climate.datasets.GeoDataset` but
    have already been pre-processed into chips.
    """

    def __init__(self, dataset, roi=None, toi=None, shuffle=False, generator=None):
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (defaults to the bounds
                of ``dataset.index``)
            toi: time of interest to sample from (defaults to the bounds of
                ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
            generator: pseudo-random number generator (PRNG) used in
                combination with shuffle
        """
        super().__init__(dataset, roi, toi)
        self.shuffle = shuffle
        self.generator = np.random.default_rng(generator)

    def __iter__(self):
        if self.shuffle:
            indices = self.generator.permutation(len(self))
        else:
            indices = range(len(self))

        for idx in indices:
            i = int(idx)
            xmin, ymin, xmax, ymax = self.index.geometry.iloc[i].bounds
            tmin, tmax = self.index.index[i].left, self.index.index[i].right
            yield slice(xmin, xmax), slice(ymin, ymax), slice(tmin, tmax)

    def __len__(self):
        return len(self.index)
