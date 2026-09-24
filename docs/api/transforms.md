# Transforms API

Auto-generated from source — a multi-backend Keras port of
[`torchgeo.transforms`](https://torchgeo.readthedocs.io/en/stable/api/transforms.html).
Image transforms expect channels-last (`... x H x W x C`) input, matching the
rest of `keras_climate` (torchgeo itself uses channels-first). See
[Data](../models/data.md) for a narrative guide.

## Band indices

::: keras_climate.transforms.indices.AppendNormalizedDifferenceIndex

::: keras_climate.transforms.indices.AppendBNDVI

::: keras_climate.transforms.indices.AppendEVI

::: keras_climate.transforms.indices.AppendGBNDVI

::: keras_climate.transforms.indices.AppendGNDVI

::: keras_climate.transforms.indices.AppendGRNDVI

::: keras_climate.transforms.indices.AppendMNDWI

::: keras_climate.transforms.indices.AppendNBR

::: keras_climate.transforms.indices.AppendNDBI

::: keras_climate.transforms.indices.AppendNDRE

::: keras_climate.transforms.indices.AppendNDSI

::: keras_climate.transforms.indices.AppendNDVI

::: keras_climate.transforms.indices.AppendNDWI

::: keras_climate.transforms.indices.AppendRBNDVI

::: keras_climate.transforms.indices.AppendSAVI

::: keras_climate.transforms.indices.AppendSWI

::: keras_climate.transforms.indices.AppendTriBandNormalizedDifferenceIndex

## Color

::: keras_climate.transforms.color.RandomGrayscale

## SAR

::: keras_climate.transforms.sar.LeeFilter

::: keras_climate.transforms.sar.lee_filter

## Spatial

::: keras_climate.transforms.spatial.SatSlideMix

## Temporal

::: keras_climate.transforms.temporal.Rearrange
