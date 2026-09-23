import numpy as np
import pandas as pd
import pytest
import shapely
from geopandas import GeoDataFrame
from pyproj import CRS

from keras_climate.datasets import GeoDataset
from keras_climate.datasets.splits import (
    random_bbox_assignment,
    random_bbox_splitting,
    random_grid_cell_assignment,
    roi_split,
    time_series_split,
)

MINT = pd.Timestamp(2021, 1, 1)
MAXT = pd.Timestamp(2021, 1, 2)


class CustomGeoDataset(GeoDataset):
    def __init__(self, bounds, crs=None, res=(1, 1)):
        crs = crs or CRS.from_epsg(4087)
        data = {"filepath": ["file.tif"] * len(bounds)}
        geometry = [shapely.box(b[0], b[2], b[1], b[3]) for b in bounds]
        index = pd.IntervalIndex.from_tuples(
            [(b[4], b[5]) for b in bounds], closed="both", name="datetime"
        )
        self.index = GeoDataFrame(data, index=index, geometry=geometry, crs=crs)
        self.res = res
        self.paths = []

    def __getitem__(self, index):
        return {"bounds": self._slice_to_tensor(index)}


@pytest.fixture
def dataset():
    bounds = [
        (0, 10, 0, 10, MINT, MAXT),
        (10, 20, 10, 20, MINT, MAXT),
        (20, 30, 20, 30, MINT, MAXT),
        (30, 40, 30, 40, MINT, MAXT),
    ]
    return CustomGeoDataset(bounds)


def test_random_bbox_assignment(dataset):
    splits = random_bbox_assignment(dataset, [0.5, 0.5], generator=np.random.default_rng(0))
    assert len(splits) == 2
    assert sum(len(s) for s in splits) == len(dataset)


def test_random_bbox_assignment_invalid_lengths(dataset):
    with pytest.raises(ValueError, match="Sum of input lengths"):
        random_bbox_assignment(dataset, [0.3, 0.3])


def test_random_bbox_assignment_negative_lengths(dataset):
    with pytest.raises(ValueError, match="greater than 0"):
        random_bbox_assignment(dataset, [1.5, -0.5])


def test_random_bbox_splitting(dataset):
    splits = random_bbox_splitting(dataset, [0.5, 0.5], generator=np.random.default_rng(0))
    assert len(splits) == 2
    for s in splits:
        assert len(s) == len(dataset)


def test_random_bbox_splitting_invalid_fractions(dataset):
    with pytest.raises(ValueError, match="Sum of input fractions"):
        random_bbox_splitting(dataset, [0.3, 0.3])


def test_random_grid_cell_assignment(dataset):
    splits = random_grid_cell_assignment(
        dataset, [0.5, 0.5], grid_size=2, generator=np.random.default_rng(0)
    )
    assert len(splits) == 2
    total_cells = sum(len(s) for s in splits)
    assert total_cells == len(dataset) * 2 * 2


def test_random_grid_cell_assignment_invalid_grid_size(dataset):
    with pytest.raises(ValueError, match="grid_size must be greater"):
        random_grid_cell_assignment(dataset, [0.5, 0.5], grid_size=1)


def test_roi_split(dataset):
    rois = [shapely.box(0, 0, 15, 15), shapely.box(15, 15, 40, 40)]
    splits = roi_split(dataset, rois)
    assert len(splits) == 2


def test_roi_split_overlapping_raises(dataset):
    rois = [shapely.box(0, 0, 20, 20), shapely.box(5, 5, 25, 25)]
    with pytest.raises(ValueError, match="can't overlap"):
        roi_split(dataset, rois)


def test_time_series_split_fractions():
    bounds = [(0, 1, 0, 1, MINT, MAXT)]
    ds = CustomGeoDataset(bounds)
    splits = time_series_split(ds, [0.5, 0.5])
    assert len(splits) == 2


def test_time_series_split_invalid_fractions():
    bounds = [(0, 1, 0, 1, MINT, MAXT)]
    ds = CustomGeoDataset(bounds)
    with pytest.raises(ValueError, match="greater than 0"):
        time_series_split(ds, [1.5, -0.5])
