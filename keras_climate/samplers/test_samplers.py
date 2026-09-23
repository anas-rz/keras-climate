import pandas as pd
import pytest
import shapely
from geopandas import GeoDataFrame
from pyproj import CRS

from keras_climate.datasets import GeoDataset
from keras_climate.samplers import (
    GriddedPatchSampler,
    GridGeoSampler,
    PreChippedGeoSampler,
    RandomBatchGeoSampler,
    RandomGeoSampler,
    RandomPatchSampler,
    RandomPeriodSampler,
    RandomTimedeltaSampler,
    RandomTimestampSampler,
    SequentialPeriodSampler,
    SequentialTimedeltaSampler,
    SequentialTimestampSampler,
    Units,
)
from keras_climate.samplers.utils import convolution_arithmetic, get_random_bounding_box, tile_to_chips

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
    return CustomGeoDataset([(0, 20, 0, 20, MINT, MAXT)])


@pytest.fixture
def multi_temporal_dataset():
    return CustomGeoDataset(
        [
            (0, 20, 0, 20, pd.Timestamp(2021, 1, 1), pd.Timestamp(2021, 1, 2)),
            (0, 20, 0, 20, pd.Timestamp(2021, 6, 1), pd.Timestamp(2021, 6, 2)),
            (0, 20, 0, 20, pd.Timestamp(2022, 1, 1), pd.Timestamp(2022, 1, 2)),
        ]
    )


class TestConvolutionArithmetic:
    def test_default_stride(self):
        assert convolution_arithmetic(10, 5) == 2

    def test_with_stride(self):
        assert convolution_arithmetic(10, 5, 2) == 4


class TestTileToChips:
    def test_basic(self):
        rows, cols = tile_to_chips((0, 0, 10, 10), (5, 5))
        assert rows == 2
        assert cols == 2


class TestGetRandomBoundingBox:
    def test_deterministic_with_seed(self):
        bbox1 = get_random_bounding_box((0, 0, 20, 20), (4, 4), (1, 1), generator=0)
        bbox2 = get_random_bounding_box((0, 0, 20, 20), (4, 4), (1, 1), generator=0)
        assert bbox1[0].start == bbox2[0].start
        assert bbox1[1].start == bbox2[1].start


class TestGriddedPatchSampler:
    def test_length(self, dataset):
        sampler = GriddedPatchSampler(dataset, size=5, stride=5)
        assert len(sampler) == len(list(sampler))

    def test_strategy(self, dataset):
        sampler = GriddedPatchSampler(dataset, size=5)
        assert sampler.strategy == "sequential"

    def test_crs_units(self, dataset):
        sampler = GriddedPatchSampler(dataset, size=5.0, units=Units.CRS)
        locs = list(sampler)
        assert len(locs) > 0


class TestRandomPatchSampler:
    def test_length(self, dataset):
        sampler = RandomPatchSampler(dataset, size=5, length=10, generator=0)
        assert len(sampler) == 10
        assert len(list(sampler)) == 10

    def test_strategy(self, dataset):
        sampler = RandomPatchSampler(dataset, size=5, length=1, generator=0)
        assert sampler.strategy == "random"

    def test_reproducible_with_seed(self, dataset):
        s1 = RandomPatchSampler(dataset, size=5, length=5, generator=0)
        s2 = RandomPatchSampler(dataset, size=5, length=5, generator=0)
        locs1 = [(sl.start, sl.stop) for x, y in s1 for sl in (x, y)]
        locs2 = [(sl.start, sl.stop) for x, y in s2 for sl in (x, y)]
        assert locs1 == locs2


class TestLegacyGridGeoSampler:
    def test_length(self, dataset):
        sampler = GridGeoSampler(dataset, size=5, stride=5)
        assert len(sampler) == len(list(sampler))
        for x, y, t in sampler:
            assert x.stop - x.start == 5
            assert y.stop - y.start == 5


class TestLegacyRandomGeoSampler:
    def test_length(self, dataset):
        sampler = RandomGeoSampler(dataset, size=5, length=6, generator=0)
        assert len(sampler) == 6
        assert len(list(sampler)) == 6


class TestPreChippedGeoSampler:
    def test_length(self, dataset):
        sampler = PreChippedGeoSampler(dataset)
        assert len(sampler) == len(dataset)

    def test_shuffle_is_permutation(self, dataset):
        ds = CustomGeoDataset(
            [
                (0, 5, 0, 5, MINT, MAXT),
                (5, 10, 5, 10, MINT, MAXT),
                (10, 15, 10, 15, MINT, MAXT),
            ]
        )
        sampler = PreChippedGeoSampler(ds, shuffle=True, generator=0)
        locs = list(sampler)
        assert len(locs) == 3


class TestRandomBatchGeoSampler:
    def test_batches(self, dataset):
        sampler = RandomBatchGeoSampler(dataset, size=5, batch_size=2, length=6, generator=0)
        batches = list(sampler)
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 2


class TestTemporalSamplers:
    def test_random_timestamp(self, multi_temporal_dataset):
        sampler = RandomTimestampSampler(multi_temporal_dataset, generator=0)
        locs = list(sampler)
        assert len(locs) == 3
        assert sampler.strategy == "random"

    def test_sequential_timestamp(self, multi_temporal_dataset):
        sampler = SequentialTimestampSampler(multi_temporal_dataset)
        locs = list(sampler)
        assert len(locs) == 3
        assert sampler.strategy == "sequential"

    def test_random_timedelta(self, multi_temporal_dataset):
        sampler = RandomTimedeltaSampler(
            multi_temporal_dataset, delta=pd.Timedelta(hours=1), length=2, generator=0
        )
        locs = list(sampler)
        assert len(locs) == 2

    def test_sequential_timedelta(self, multi_temporal_dataset):
        sampler = SequentialTimedeltaSampler(
            multi_temporal_dataset, delta=pd.Timedelta(hours=6)
        )
        locs = list(sampler)
        assert len(locs) > 0

    def test_random_period(self, multi_temporal_dataset):
        sampler = RandomPeriodSampler(
            multi_temporal_dataset, freq="D", length=2, generator=0
        )
        locs = list(sampler)
        assert len(locs) == 2

    def test_sequential_period(self, multi_temporal_dataset):
        sampler = SequentialPeriodSampler(multi_temporal_dataset, freq="D")
        locs = list(sampler)
        assert len(locs) >= 3
