"""Multi-backend Keras port of torchgeo.samplers.

Samplers operate purely on geospatial indices (pandas/geopandas/shapely) and
yield plain Python slices, so they have no dependency on the active Keras
tensor backend at all.
"""

from .base import GeoSampler, SpatialSampler, SpatioTemporalSampler, TemporalSampler
from .batch import BatchGeoSampler, RandomBatchGeoSampler
from .constants import Units
from .single import GridGeoSampler, PreChippedGeoSampler, RandomGeoSampler
from .spatial import GriddedPatchSampler, RandomPatchSampler
from .temporal import (
    RandomPeriodSampler,
    RandomTimedeltaSampler,
    RandomTimestampSampler,
    SequentialPeriodSampler,
    SequentialTimedeltaSampler,
    SequentialTimestampSampler,
)
from .utils import get_random_bounding_box, tile_to_chips

__all__ = (
    "BatchGeoSampler",
    "GeoSampler",
    "GridGeoSampler",
    "GriddedPatchSampler",
    "PreChippedGeoSampler",
    "RandomBatchGeoSampler",
    "RandomGeoSampler",
    "RandomPatchSampler",
    "RandomPeriodSampler",
    "RandomTimedeltaSampler",
    "RandomTimestampSampler",
    "SequentialPeriodSampler",
    "SequentialTimedeltaSampler",
    "SequentialTimestampSampler",
    "SpatialSampler",
    "SpatioTemporalSampler",
    "TemporalSampler",
    "Units",
    "get_random_bounding_box",
    "tile_to_chips",
)
