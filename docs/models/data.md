# Data: Datasets, Samplers, Transforms, Losses

`keras_climate.datasets` / `keras_climate.samplers` / `keras_climate.transforms`
/ `keras_climate.losses` are a multi-backend Keras port of
[`torchgeo`](https://torchgeo.readthedocs.io/)'s data-loading stack — the
same geospatial dataset framework and catalog of 190+ concrete datasets
(EuroSAT, So2Sat, Sentinel, SpaceNet, Copernicus-Bench, ...), reworked so
sample values are Keras tensors (`keras.ops.convert_to_tensor`) instead of
`torch.Tensor`s, and so datasets work unchanged under the TensorFlow, JAX,
and PyTorch Keras 3 backends.

!!! important "Channels-last, not channels-first"
    Unlike torchgeo (`C x H x W`), image/mask samples here use the
    **channels-last** layout (`H x W x C`) to match the rest of
    `keras_climate` and Keras's own default `Conv2D` data format.

## Two kinds of dataset

- **`NonGeoDataset`** (and `NonGeoClassificationDataset`) — a plain
  index-based dataset with no geospatial metadata, e.g. a folder of
  pre-chipped tiles. Index directly with `dataset[i]`, same as any
  `torch.utils.data.Dataset`-style class.
- **`GeoDataset`** (and `RasterDataset` / `VectorDataset` / `XarrayDataset`)
  — a dataset backed by georeferenced files (rasters, vector layers,
  xarray-backed cubes), indexed by a `BoundingBox` **query** rather than a
  plain integer, so a single dataset can span an arbitrary tiled/mosaicked
  area of interest. `IntersectionDataset`/`UnionDataset` compose two
  `GeoDataset`s (e.g. imagery + a label raster) by their spatiotemporal
  extent.

Every sample is a `dict` (typically with `"image"` and, for labeled
datasets, `"mask"` or `"label"` keys) of Keras tensors.

## Classification example (`NonGeoDataset`)

```python
from keras_climate.datasets import EuroSAT

train_ds = EuroSAT(root="data/eurosat", split="train", download=True)
sample = train_ds[0]
print(sample["image"].shape, sample["label"])  # (64, 64, 13), a scalar int64 tensor
print(train_ds.classes)
```

## Tiled/geospatial example (`GeoDataset` + samplers)

`keras_climate.samplers` yields `BoundingBox` queries over a `GeoDataset`'s
spatiotemporal index — purely index arithmetic (pandas/geopandas/shapely),
with no dependency on the active Keras backend at all:

```python
from keras_climate.samplers import RandomGeoSampler
from keras_climate.datasets import stack_samples

sampler = RandomGeoSampler(some_raster_dataset, size=256, length=1000)

samples = [some_raster_dataset[bbox] for bbox in list(sampler)[:8]]
batch = stack_samples(samples)  # {"image": (8, 256, 256, C), ...}
```

`GridGeoSampler` tiles a region of interest exhaustively (e.g. for
inference over a full scene); `RandomBatchGeoSampler` yields whole batches
of bounding boxes at once. See [Samplers API](../api/samplers.md) for the
full list, including temporal samplers (`RandomTimestampSampler`, ...) for
`SpatioTemporalSampler`-backed datasets.

## Feeding `model.fit()`

Neither `keras_climate.datasets` nor `keras_climate.samplers` ships a
`keras.utils.PyDataset`/`tf.data` wrapper — you write a thin one (or a
generator) around the sample-fetching loop above, same as you would for any
other custom data source:

```python
import numpy as np
import keras

class MySequence(keras.utils.PyDataset):
    def __init__(self, dataset, batch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.dataset, self.batch_size = dataset, batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        samples = [self.dataset[i] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        batch = stack_samples(samples)
        return keras.ops.convert_to_numpy(batch["image"]), keras.ops.convert_to_numpy(batch["label"])

model.fit(MySequence(train_ds), epochs=10)
```

See the [finetuning example notebook](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/03_finetune_pretrained_model.ipynb)
for a complete, runnable version of this pattern.

## Transforms

`keras_climate.transforms` operate on the same `dict` samples, expecting
channels-last input, and are meant to be composed inside a dataset's own
`transforms=` callback:

```python
from keras_climate.transforms import AppendNDVI

# appends an NDVI band computed from the given red/NIR band indices
ndvi = AppendNDVI(index_red=3, index_nir=7)
train_ds = EuroSAT(root="data/eurosat", split="train", transforms=ndvi, download=True)
```

Other families: band-index transforms for other indices (`AppendNDWI`,
`AppendNBR`, ...), `LeeFilter`/`lee_filter` for SAR speckle reduction,
`SatSlideMix` for a segmentation-aware mixing augmentation, and `Rearrange`
for axis reordering. See [Transforms API](../api/transforms.md).

## Losses

`keras_climate.losses` ships two loss functions with no direct Keras
built-in equivalent, both from crop-type/time-series classification
literature: `EarlyRewardLoss` (the ELECTS loss for early classification of
time series — rewards a correct *and early* stopping decision) and
`QRLoss`/`RQLoss` (a forward/backward pair for calibrating class
probabilities against predictions). See [Losses API](../api/losses.md).

## Full dataset catalog

See [Datasets API](../api/datasets.md) for every dataset class
(190+, grouped by SpaceNet / Copernicus-Bench / the general catalog), each
with parameters and citation info pulled directly from its docstring.
