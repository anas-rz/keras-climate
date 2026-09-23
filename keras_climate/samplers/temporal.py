"""Temporal sampling routines (ported from torchgeo.samplers.temporal)."""

import numpy as np
import pandas as pd
from pandas import Interval, Period, Timestamp

from .base import TemporalSampler
from .utils import convolution_arithmetic


class RandomTimestampSampler(TemporalSampler):
    """Random sampling of single images (:class:`~pandas.Timestamp`)."""

    strategy = "random"

    def __init__(self, dataset, *, toi=None, generator=None):
        """Initialize a new RandomTimestampSampler instance.

        Args:
            dataset: Dataset to sample from.
            toi: Time of interest to sample from (defaults to the bounds of
                ``dataset.index``).
            generator: Pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, toi=toi)
        self.generator = np.random.default_rng(generator)

    def _iter_subset(self, location=(slice(None), slice(None))):
        intervals = self._init_subset(location)

        # Ensure time intervals are unique
        # Allows all intervals to be equally weighted, regardless of # locations
        intervals = pd.unique(intervals)

        intervals = intervals.to_series().sample(frac=1, random_state=self.generator)

        x, y = location
        for interval in intervals:
            t = slice(interval.left, interval.right)
            yield x, y, t


class SequentialTimestampSampler(TemporalSampler):
    """Sequential sampling of single images (:class:`~pandas.Timestamp`)."""

    strategy = "sequential"

    def _iter_subset(self, location=(slice(None), slice(None))):
        intervals = self._init_subset(location)

        # Ensure time intervals are unique to avoid repeats
        intervals = sorted(pd.unique(intervals))

        x, y = location
        for interval in intervals:
            t = slice(interval.left, interval.right)
            yield x, y, t


class RandomTimedeltaSampler(TemporalSampler):
    """Random sampling of fixed-length sliding windows (:class:`~pandas.Timedelta`)."""

    strategy = "random"

    def __init__(self, dataset, *, delta, length=None, toi=None, generator=None):
        """Initialize a new RandomTimedeltaSampler instance.

        Args:
            dataset: Dataset to sample from.
            delta: Duration of time.
            length: Number of random samples to draw per epoch (defaults to
                approximately the maximal number of non-overlapping
                intervals that could be sampled from the dataset).
            toi: Time of interest to sample from (defaults to the bounds of
                ``dataset.index``).
            generator: Pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, toi=toi)
        self.delta = delta
        left = self.index.index.left.min()
        right = self.index.index.right.max()
        self._length = length or convolution_arithmetic(right - left, delta)
        self.generator = np.random.default_rng(generator)

    def _iter_subset(self, location=(slice(None), slice(None))):
        intervals = self._init_subset(location)

        left = intervals.left.min()
        right = intervals.right.max() - self.delta

        i = 0
        x, y = location
        while i < len(self):
            ts = self.generator.uniform(left.timestamp(), right.timestamp())
            tmin = Timestamp.fromtimestamp(ts)
            tmax = tmin + self.delta
            interval = Interval(tmin, tmax)
            if intervals.overlaps(interval).any():
                t = slice(interval.left, interval.right)
                yield x, y, t
                i += 1


class SequentialTimedeltaSampler(TemporalSampler):
    """Sequential sampling of fixed-length sliding windows (:class:`~pandas.Timedelta`)."""

    strategy = "sequential"

    def __init__(self, dataset, *, delta, stride=None, toi=None):
        """Initialize a new SequentialTimedeltaSampler instance.

        Args:
            dataset: Dataset to sample from.
            delta: Duration of time.
            stride: Duration to skip between each sample (defaults to
                *delta*).
            toi: Time of interest to sample from (defaults to the bounds of
                ``dataset.index``).
        """
        super().__init__(dataset, toi=toi)
        self.delta = delta
        self.stride = stride or delta

    def _iter_subset(self, location=(slice(None), slice(None))):
        intervals = self._init_subset(location)

        left = intervals.left.min()
        right = intervals.right.max()
        length = convolution_arithmetic(right - left, self.delta, self.stride)

        x, y = location
        for _ in range(length):
            interval = Interval(left, left + self.delta)
            if intervals.overlaps(interval).any():
                t = slice(interval.left, interval.right)
                yield x, y, t
            left += self.stride


class RandomPeriodSampler(TemporalSampler):
    """Random sampling of fixed-length fixed windows (:class:`~pandas.Period`)."""

    strategy = "random"

    def __init__(self, dataset, *, freq, length=None, toi=None, generator=None):
        """Initialize a new RandomPeriodSampler instance.

        Args:
            dataset: Dataset to sample from.
            freq: Temporal frequencies to sample. Accepts any valid pandas
                period alias.
            length: Number of random samples to draw per epoch (defaults to
                approximately the maximal number of non-overlapping periods
                that could be sampled from the dataset).
            toi: Time of interest to sample from (defaults to the bounds of
                ``dataset.index``).
            generator: Pseudo-random number generator (PRNG).
        """
        super().__init__(dataset, toi=toi)
        self.freq = freq
        left = self.index.index.left.min()
        right = self.index.index.right.max()
        period = Period(left, freq=freq)
        window = period.end_time - period.start_time
        self._length = length or convolution_arithmetic(right - left, window)
        self.generator = np.random.default_rng(generator)

    def _iter_subset(self, location=(slice(None), slice(None))):
        intervals = self._init_subset(location)

        left = intervals.left.min()
        right = intervals.right.max()

        # Expand to full period to support balanced sampling
        left = Period(left, freq=self.freq).start_time
        right = Period(right, freq=self.freq).end_time

        i = 0
        x, y = location
        while i < len(self):
            ts = self.generator.uniform(left.timestamp(), right.timestamp())
            period = Period(Timestamp.fromtimestamp(ts), freq=self.freq)
            interval = Interval(period.start_time, period.end_time)
            if intervals.overlaps(interval).any():
                t = slice(interval.left, interval.right)
                yield x, y, t
                i += 1


class SequentialPeriodSampler(TemporalSampler):
    """Sequential sampling of fixed-length fixed windows (:class:`~pandas.Period`)."""

    strategy = "sequential"

    def __init__(self, dataset, *, freq, toi=None):
        """Initialize a new SequentialPeriodSampler instance.

        Args:
            dataset: Dataset to sample from.
            freq: Temporal frequencies to sample. Accepts any valid pandas
                period alias.
            toi: Time of interest to sample from (defaults to the bounds of
                ``dataset.index``).
        """
        super().__init__(dataset, toi=toi)
        self.freq = freq

    def _iter_subset(self, location=(slice(None), slice(None))):
        intervals = self._init_subset(location)

        left = intervals.left.min()
        right = intervals.right.max()

        x, y = location
        while left < right:
            period = Period(left, freq=self.freq)
            interval = Interval(period.start_time, period.end_time)
            if intervals.overlaps(interval).any():
                t = slice(interval.left, interval.right)
                yield x, y, t
            left = interval.mid + (period.end_time - period.start_time)
