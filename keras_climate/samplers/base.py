"""Sampler base classes (ported from torchgeo.samplers.base)."""

import abc
import warnings
from abc import ABC

import numpy as np
import shapely
import shapely.plotting
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely import MultiPolygon, Polygon

from .utils import prism


class GeoSampler(ABC):
    """Abstract base class for sampling from a :class:`~keras_climate.datasets.GeoDataset`.

    Returns a ``GeoSlice`` that can uniquely index any
    :class:`~keras_climate.datasets.GeoDataset`.
    """

    _length: int

    @abc.abstractmethod
    def __iter__(self):
        """Iterate over generated sample locations for each epoch."""

    def __len__(self):
        if not hasattr(self, "_length"):
            self._length = sum(1 for _ in self)

        return self._length


class SpatialSampler(GeoSampler):
    """Abstract base class for all spatial sampling strategies."""

    @property
    @abc.abstractmethod
    def strategy(self):
        """Sampling strategy: one of 'random' or 'sequential'.

        This distinction only matters when combining samplers via
        :class:`SpatioTemporalSampler`, where either a zip (random) or
        product (sequential) of all sample locations is taken during each
        epoch.
        """

    def __init__(self, dataset, *, roi=None):
        """Initialize a new SpatialSampler instance.

        Args:
            dataset: Dataset to sample from.
            roi: Region of interest to sample from (defaults to the bounds
                of ``dataset.index``).
        """
        # Create one single MultiPolygon of all objects
        # Allows all locations to be equally weighted, regardless of # time stamps
        self.geometry = dataset.index.geometry.union_all()
        self.bounds = self.geometry.bounds
        self.res = dataset.res

        if roi is not None:
            self.geometry &= roi

    @abc.abstractmethod
    def __iter__(self):
        """Iterate over generated sample locations for each epoch."""

    def __matmul__(self, other):
        """Compute the product of two samplers."""
        return SpatioTemporalSampler(self, other)

    def plot(self):
        """Plot a visualization of the sampling strategy."""
        geometry = self.geometry
        assert isinstance(geometry, (Polygon, MultiPolygon))
        xmin, ymin, xmax, ymax = geometry.bounds

        fig, ax = plt.subplots()
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("equal")

        def init_func():
            return shapely.plotting.plot_polygon(geometry, ax=ax)

        def func(index):
            x, y = index
            xy = (x.start, y.start)
            width = x.stop - x.start
            height = y.stop - y.start
            patch = Rectangle(xy, width, height, color="tab:orange", alpha=0.3)
            ax.add_patch(patch)
            return [patch]

        return FuncAnimation(fig, func=func, frames=self, init_func=init_func)


class TemporalSampler(GeoSampler):
    """Abstract base class for all temporal sampling strategies."""

    @property
    @abc.abstractmethod
    def strategy(self):
        """Sampling strategy: one of 'random' or 'sequential'."""

    def __init__(self, dataset, *, toi=None):
        """Initialize a new TemporalSampler instance.

        Args:
            dataset: Dataset to sample from.
            toi: Time of interest to sample from (defaults to the bounds of
                ``dataset.index``).
        """
        self.index = dataset.index

        if toi is not None:
            import pandas as pd

            tmin = np.maximum(toi.left.to_datetime64(), self.index.index.left)
            tmax = np.minimum(toi.right.to_datetime64(), self.index.index.right)
            valid = tmax >= tmin
            tmin = tmin[valid]
            tmax = tmax[valid]
            self.index = self.index[valid]
            self.index.index = pd.IntervalIndex.from_arrays(
                tmin, tmax, closed="both", name="datetime"
            )

    def __iter__(self):
        yield from self._iter_subset()

    def _init_subset(self, location=(slice(None), slice(None))):
        """Narrow down index to a specific location."""
        index = self.index

        # Since this only occurs in combination with a SpatialSampler, x and y are
        # guaranteed to have start and stop, and t is guaranteed to be empty
        x, y = location
        index = index.cx[x.start : x.stop, y.start : y.stop]

        return index.index

    @abc.abstractmethod
    def _iter_subset(self, location=(slice(None), slice(None))):
        """Iterate over generated sample locations for each epoch."""

    def plot(self):
        """Plot a visualization of the sampling strategy."""
        tmin = self.index.index.left.min()
        tmax = self.index.index.right.max()

        fig, ax = plt.subplots()
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel("t")
        ax.set_xlim(tmin, tmax)
        ax.yaxis.set_visible(False)
        ax.spines[["left", "top", "right"]].set_visible(False)
        fig.autofmt_xdate()

        def init_func():
            patches = []
            for interval in self.index.index:
                patch = ax.axvspan(
                    interval.left, interval.right, color="tab:blue", alpha=0.3
                )
                patches.append(patch)
            return patches

        def func(index):
            _, _, t = index
            patch = ax.axvspan(t.start, t.stop, color="tab:orange", alpha=0.3)
            ax.add_patch(patch)
            return [patch]

        return FuncAnimation(fig, func=func, frames=self, init_func=init_func)


class SpatioTemporalSampler(GeoSampler):
    """Product of a spatial and a temporal sampler."""

    def __init__(self, spatial_sampler, temporal_sampler):
        """Initialize a new SpatioTemporalSampler instance.

        Args:
            spatial_sampler: A spatial sampling strategy.
            temporal_sampler: A temporal sampling strategy.
        """
        self.spatial_sampler = spatial_sampler
        self.temporal_sampler = temporal_sampler

        if (
            self.spatial_sampler.strategy == "random"
            and self.temporal_sampler.strategy == "sequential"
        ):
            msg = "random_sampler @ sequential_sampler may result in a different "
            msg += "number of samples per epoch if different random locations have "
            msg += "a different number of timestamps"
            warnings.warn(msg, UserWarning)

    def __len__(self):
        if not hasattr(self, "_length"):
            spatial_strategy = self.spatial_sampler.strategy
            temporal_strategy = self.temporal_sampler.strategy
            if spatial_strategy == "random" and temporal_strategy == "random":
                self._length = len(self.spatial_sampler)
            elif spatial_strategy == "sequential" and temporal_strategy == "random":
                self._length = len(self.spatial_sampler) * len(self.temporal_sampler)
            else:
                self._length = super().__len__()
        return self._length

    def __iter__(self):
        spatial_strategy = self.spatial_sampler.strategy
        temporal_strategy = self.temporal_sampler.strategy

        if spatial_strategy == "random" and temporal_strategy == "random":
            spatial_iter = iter(self.spatial_sampler)
            for _ in range(len(self.spatial_sampler)):
                location = next(spatial_iter)
                yield next(self.temporal_sampler._iter_subset(location))
        elif spatial_strategy == "sequential" and temporal_strategy == "sequential":
            for location in self.spatial_sampler:
                for index in self.temporal_sampler._iter_subset(location):
                    yield index
        elif spatial_strategy == "random" and temporal_strategy == "sequential":
            spatial_iter = iter(self.spatial_sampler)
            for _ in range(len(self.spatial_sampler)):
                location = next(spatial_iter)
                for index in self.temporal_sampler._iter_subset(location):
                    yield index
        elif spatial_strategy == "sequential" and temporal_strategy == "random":
            for location in self.spatial_sampler:
                for _ in range(len(self.temporal_sampler)):
                    yield next(self.temporal_sampler._iter_subset(location))

    def plot(self):
        """Plot a visualization of the sampling strategy."""
        spatial = self.spatial_sampler
        temporal = self.temporal_sampler

        xmin, ymin, xmax, ymax = spatial.geometry.bounds
        tmin = temporal.index.index.left.min().timestamp()
        tmax = temporal.index.index.right.max().timestamp()

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_title(f"{spatial.__class__.__name__} @ {temporal.__class__.__name__}")
        ax.set(xlabel="x", ylabel="y", zlabel="t")
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[tmin, tmax])
        ax.set_aspect("equalxy")

        def init_func():
            verts = []
            for index, data in temporal.index.iterrows():
                x, y = data["geometry"].exterior.coords.xy
                tmin = index.left.timestamp()
                tmax = index.right.timestamp()
                t = np.array([tmin, tmax])
                verts.extend(prism(x, y, t))
            poly = Poly3DCollection(verts, color="tab:blue", alpha=0.3)
            ax.add_collection3d(poly)
            return (poly,)

        def func(index):
            x = np.array(
                [
                    index[0].start,
                    index[0].start,
                    index[0].stop,
                    index[0].stop,
                    index[0].start,
                ]
            )
            y = np.array(
                [
                    index[1].start,
                    index[1].stop,
                    index[1].stop,
                    index[1].start,
                    index[1].start,
                ]
            )
            t = np.array([index[2].start.timestamp(), index[2].stop.timestamp()])
            verts = prism(x, y, t)
            poly = Poly3DCollection(verts, color="tab:orange", alpha=0.3)
            ax.add_collection3d(poly)
            return (poly,)

        return FuncAnimation(fig, func=func, frames=self, init_func=init_func)
