# Samplers API

Auto-generated from source — a multi-backend Keras port of
[`torchgeo.samplers`](https://torchgeo.readthedocs.io/en/stable/api/samplers.html).
Samplers operate purely on geospatial indices (pandas/geopandas/shapely) and
yield plain Python slices, so they have no dependency on the active Keras
tensor backend at all. See [Data](../models/data.md) for a narrative guide.

## Base classes

::: keras_climate.samplers.base.GeoSampler

::: keras_climate.samplers.base.SpatialSampler

::: keras_climate.samplers.base.SpatioTemporalSampler

::: keras_climate.samplers.base.TemporalSampler

::: keras_climate.samplers.batch.BatchGeoSampler

## Single-item samplers

::: keras_climate.samplers.single.GridGeoSampler

::: keras_climate.samplers.single.PreChippedGeoSampler

::: keras_climate.samplers.single.RandomGeoSampler

::: keras_climate.samplers.spatial.GriddedPatchSampler

::: keras_climate.samplers.spatial.RandomPatchSampler

## Batch samplers

::: keras_climate.samplers.batch.RandomBatchGeoSampler

## Temporal samplers

::: keras_climate.samplers.temporal.RandomPeriodSampler

::: keras_climate.samplers.temporal.RandomTimedeltaSampler

::: keras_climate.samplers.temporal.RandomTimestampSampler

::: keras_climate.samplers.temporal.SequentialPeriodSampler

::: keras_climate.samplers.temporal.SequentialTimedeltaSampler

::: keras_climate.samplers.temporal.SequentialTimestampSampler

## Utilities

::: keras_climate.samplers.constants.Units

::: keras_climate.samplers.utils.get_random_bounding_box

::: keras_climate.samplers.utils.tile_to_chips
