"""Dataset splitting utilities (ported from torchgeo.datasets.splits)."""

import itertools
from copy import deepcopy
from itertools import accumulate
from math import floor, isclose

import geopandas
import numpy as np
import pandas as pd
import shapely
import shapely.ops
from geopandas import GeoDataFrame
from shapely import LineString


def _fractions_to_lengths(fractions, total):
    """Utility to divide a number into a list of integers according to fractions."""
    lengths = [floor(frac * total) for frac in fractions]
    remainder = int(total - sum(lengths))
    for i in range(remainder):
        idx_to_add_at = i % len(lengths)
        lengths[idx_to_add_at] += 1
    return lengths


def random_bbox_assignment(dataset, lengths, generator=None):
    """Split a GeoDataset randomly assigning its index's objects."""
    rng = np.random.default_rng() if generator is None else generator

    if not (isclose(sum(lengths), 1) or isclose(sum(lengths), len(dataset))):
        raise ValueError(
            "Sum of input lengths must equal 1 or the length of dataset's index."
        )

    if any(n <= 0 for n in lengths):
        raise ValueError("All items in input lengths must be greater than 0.")

    if isclose(sum(lengths), 1):
        lengths = _fractions_to_lengths(lengths, len(dataset))

    indices = rng.permutation(sum(lengths))

    new_datasets = []
    for offset, length in zip(itertools.accumulate(lengths), lengths):
        ds = deepcopy(dataset)
        ds.index = dataset.index.iloc[indices[offset - length : offset]]
        new_datasets.append(ds)

    return new_datasets


def random_bbox_splitting(dataset, fractions, generator=None):
    """Split a GeoDataset randomly splitting its index's objects."""
    rng = np.random.default_rng() if generator is None else generator

    if not isclose(sum(fractions), 1):
        raise ValueError("Sum of input fractions must equal 1.")

    if any(n <= 0 for n in fractions):
        raise ValueError("All items in input fractions must be greater than 0.")

    i_geom = dataset.index.columns.get_loc("geometry")
    new_datasets = [deepcopy(dataset) for _ in fractions]

    for i in range(len(dataset)):
        geometry_remaining = dataset.index.geometry.iloc[i]
        fraction_remaining = 1.0

        horizontal, flip = rng.integers(0, 2, (2,))
        for j, fraction in enumerate(fractions):
            if isclose(fraction_remaining, fraction):
                new_geometry = geometry_remaining
            else:
                minx, miny, maxx, maxy = geometry_remaining.bounds

                if flip:
                    frac = fraction_remaining - fraction
                else:
                    frac = fraction

                if horizontal:
                    splity = miny + (maxy - miny) * frac / fraction_remaining
                    line = LineString([(minx, splity), (maxx, splity)])
                else:
                    splitx = minx + (maxx - minx) * frac / fraction_remaining
                    line = LineString([(splitx, miny), (splitx, maxy)])

                geom1, geom2 = shapely.ops.split(geometry_remaining, line).geoms
                if horizontal:
                    if flip:
                        if geom1.centroid.y < splity:
                            geometry_remaining, new_geometry = geom1, geom2
                        else:
                            new_geometry, geometry_remaining = geom1, geom2
                    else:
                        if geom1.centroid.y < splity:
                            new_geometry, geometry_remaining = geom1, geom2
                        else:
                            geometry_remaining, new_geometry = geom1, geom2
                else:
                    if flip:
                        if geom1.centroid.x < splitx:
                            geometry_remaining, new_geometry = geom1, geom2
                        else:
                            new_geometry, geometry_remaining = geom1, geom2
                    else:
                        if geom1.centroid.x < splitx:
                            new_geometry, geometry_remaining = geom1, geom2
                        else:
                            geometry_remaining, new_geometry = geom1, geom2
            new_datasets[j].index.iloc[i, i_geom] = new_geometry

            fraction_remaining -= fraction
            horizontal = not horizontal

    return new_datasets


def random_grid_cell_assignment(dataset, fractions, grid_size=6, generator=None):
    """Overlays a grid over a GeoDataset and randomly assigns cells to new GeoDatasets."""
    rng = np.random.default_rng() if generator is None else generator

    if not isclose(sum(fractions), 1):
        raise ValueError("Sum of input fractions must equal 1.")

    if any(n <= 0 for n in fractions):
        raise ValueError("All items in input fractions must be greater than 0.")

    if grid_size < 2:
        raise ValueError("Input grid_size must be greater than 1.")

    left = []
    right = []
    rows = []
    geometry = []
    for index, row in dataset.index.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds

        stridex = (maxx - minx) / grid_size
        stridey = (maxy - miny) / grid_size

        for x in range(grid_size):
            for y in range(grid_size):
                geom = shapely.box(
                    minx + x * stridex,
                    miny + y * stridey,
                    minx + (x + 1) * stridex,
                    miny + (y + 1) * stridey,
                )
                if geom := shapely.intersection(row.geometry, geom):
                    left.append(index.left)
                    right.append(index.right)
                    rows.append(row)
                    geometry.append(geom)

    lengths = _fractions_to_lengths(fractions, len(rows))

    indexes_sr = pd.IntervalIndex.from_arrays(left, right, closed="both", name="datetime")
    rows_df = pd.DataFrame(rows)
    geometry_sr = pd.Series(geometry)

    indices = rng.permutation(len(rows))

    new_datasets = []
    for offset, length in zip(itertools.accumulate(lengths), lengths):
        ds = deepcopy(dataset)
        idx = indices[offset - length : offset].tolist()
        ds.index = GeoDataFrame(
            data=rows_df.iloc[idx].values,
            index=indexes_sr[idx],
            geometry=geometry_sr[idx].values,
            crs=dataset.crs,
        )
        new_datasets.append(ds)

    return new_datasets


def roi_split(dataset, rois):
    """Split a GeoDataset intersecting it with a ROI for each desired new GeoDataset."""
    new_datasets = []
    for i, roi in enumerate(rois):
        if any(
            shapely.intersects(roi, x) and not shapely.touches(roi, x)
            for x in rois[i + 1 :]
        ):
            raise ValueError("ROIs in input rois can't overlap.")

        ds = deepcopy(dataset)
        ds.index = geopandas.clip(dataset.index, roi)
        new_datasets.append(ds)

    return new_datasets


def time_series_split(dataset, lengths):
    """Split a GeoDataset on its time dimension to create non-overlapping GeoDatasets."""
    _, _, t = dataset.bounds

    totalt = t.stop - t.start

    if all(isinstance(x, (int, float)) for x in lengths):
        if any(n <= 0 for n in lengths):
            raise ValueError("All items in input lengths must be greater than 0.")

        if not isclose(sum(lengths), 1):
            raise ValueError(
                "Sum of input lengths must equal 1 or the dataset's time length."
            )

        lengths = [totalt * f for f in lengths]

    if all(isinstance(x, pd.Timedelta) for x in lengths):
        lengths = [
            pd.Interval(t.start + offset - length, t.start + offset, closed="neither")
            for offset, length in zip(accumulate(lengths), lengths)
        ]

    _totalt = pd.Timedelta(0)
    new_datasets = []
    for i, interval in enumerate(lengths):
        start = interval.left
        end = interval.right

        offset = (
            pd.Timedelta(0) if i == len(lengths) - 1 else pd.Timedelta(1, unit="us")
        )

        if start < t.start or end > t.stop:
            raise ValueError(
                "Pairs of timestamps in lengths can't be out of dataset's time bounds."
            )

        for other in lengths:
            left = other.left
            right = other.right
            if start < left < end or start < right < end:
                raise ValueError("Pairs of timestamps in lengths can't overlap.")

        ds = deepcopy(dataset)
        ds.index = dataset.index.iloc[dataset.index.index.overlaps(interval)]
        new_index = []
        for xy in ds.index.index:
            left = xy.left
            right = xy.right
            left = max(start, left)
            right = min(end - offset, right - offset)
            new_index.append(pd.Interval(left, right, closed="neither"))
        ds.index.index = pd.IntervalIndex(new_index, closed="neither", name="datetime")
        new_datasets.append(ds)
        _totalt += end - start

    if not _totalt == totalt:
        raise ValueError(
            "Pairs of timestamps in lengths must cover dataset's time bounds."
        )

    return new_datasets
