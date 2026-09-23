"""Deprecated batch geo samplers (ported from torchgeo.samplers.batch)."""

import abc

import numpy as np
import pandas as pd
import shapely

from .constants import Units
from .utils import _to_tuple, get_random_bounding_box, tile_to_chips


class BatchGeoSampler(abc.ABC):
    """Abstract base class for sampling batches from a :class:`~keras_climate.datasets.GeoDataset`."""

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
        """Return a batch of indices of a dataset."""


class RandomBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    .. deprecated:: Use :class:`~keras_climate.samplers.RandomPatchSampler` instead.

    This is particularly useful during training when you want to maximize
    the size of the dataset and return as many random chips as possible.
    Note that randomly sampled chips may overlap.
    """

    def __init__(
        self,
        dataset,
        size,
        batch_size,
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
            batch_size: number of samples per batch
            length: number of samples per epoch (defaults to approximately
                the maximal number of non-overlapping chips of size ``size``
                that could be sampled from the dataset)
            roi: region of interest to sample from (defaults to the bounds
                of ``dataset.index``)
            toi: time of interest to sample from (defaults to the bounds of
                ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
            generator: pseudo-random number generator (PRNG)
        """
        super().__init__(dataset, roi, toi)
        self.size = _to_tuple(size)
        self.generator = np.random.default_rng(generator)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])

        self.batch_size = batch_size
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

            # Choose random indices within that tile
            batch = []
            for _ in range(self.batch_size):
                bounding_box = get_random_bounding_box(
                    bounds, self.size, self.res, self.generator
                )
                batch.append((*bounding_box, slice(interval.left, interval.right)))

            yield batch

    def __len__(self):
        return self.length // self.batch_size
