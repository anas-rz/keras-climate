# Finetune a pretrained Sentinel-2 backbone on EuroSAT

**Backend:** Multi-Backend (TensorFlow / JAX / PyTorch)
**Model:** `ssl4eo_resnet50_moco` (ResNet-50)
**Dataset:** `EuroSAT100` (13-band Sentinel-2)
**Description:** Finetune a self-supervised pretrained Sentinel-2 backbone
for land-cover classification — freeze the backbone, train a new head,
then unfreeze and finetune end-to-end at a lower learning rate.

[View in Colab](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/03_finetune_pretrained_model.ipynb){ .md-button }
[GitHub source](https://github.com/anas-rz/keras-climate/blob/main/examples/03_finetune_pretrained_model.ipynb){ .md-button }

---

This walkthrough finetunes `ssl4eo_resnet50_moco` — a ResNet-50 backbone
self-supervised pretrained (MoCo v2) on 13-band Sentinel-2 imagery, via
TorchGeo's clean re-export of the official `zhu-xlab/SSL4EO-S12` checkpoint
— for land-cover classification on **EuroSAT** (also 13-band Sentinel-2, so
the band convention lines up exactly with what the backbone was pretrained
on).

We use `EuroSAT100`, a 100-image subset intended for tutorials, so this
runs quickly end-to-end even on a free Colab GPU. Swap in the full
`EuroSAT` (or your own `keras_climate.datasets`/`RasterDataset` subclass)
for real training.

The pattern here — load a pretrained backbone, attach a new head, train
the head with the backbone frozen, then unfreeze and finetune end-to-end
at a lower learning rate — generalizes to any of the encoder-only
pretrained models in [Pretrained Weights](../pretrained-weights.md) (e.g.
`prithvi_eo_100m`, `croma_base`, `scalemae_vitlarge_fmow`).

## Install

```bash
# keras_climate itself, plus the geospatial stack keras_climate.datasets needs
# (rasterio/geopandas/etc. aren't preinstalled on Colab).
pip install -q keras-climate rasterio geopandas shapely pyproj rtree
```

```python
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers
import numpy as np
```

## Load the pretrained backbone

`ssl4eo_resnet50_moco()` downloads the real checkpoint (cached locally so
it's only fetched once), ports it through `WeightConverter`, and returns
the ResNet-50 **backbone only** (final feature map, no classification head
— exactly what we want to attach a new head to).

```python
from keras_climate.weights.pretrained import ssl4eo_resnet50_moco

IMAGE_SIZE = 224  # the size the backbone was pretrained at

backbone, report = ssl4eo_resnet50_moco(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 13))

n_matched = len(report["matched"])
n_total = n_matched + len(report["missing_in_source"])
print(f"Converted {n_matched}/{n_total} weights from the real checkpoint.")
print("backbone output shape:", backbone.output_shape)
```

## Attach a classification head

```python
from keras_climate.datasets import EuroSAT100

# Peek at the dataset just to get the class count before building the head.
_peek = EuroSAT100(root="data/eurosat100", split="train", download=True)
NUM_CLASSES = len(_peek.classes)
print("classes:", _peek.classes)

x = layers.GlobalAveragePooling2D(name="gap")(backbone.output)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
finetune_model = keras.Model(backbone.input, outputs, name="ssl4eo_eurosat")

# Freeze the pretrained backbone for phase 1 (head-only training below).
# `backbone` and `finetune_model` share the same underlying layer objects,
# so this freezes those layers everywhere they're used.
backbone.trainable = False
finetune_model.summary()
```

## Load EuroSAT and wrap it for `model.fit()`

`keras_climate.datasets` returns one sample at a time as a
`{"image": ..., "label": ...}` dict (the `keras_climate.datasets`/
`samplers` framework is a Keras port of `torchgeo.datasets`/`samplers` —
see the [Data](../models/data.md) guide for the full geospatial-sampling
workflow). We batch and resize with a small `keras.utils.PyDataset` here
since EuroSAT's 64x64 tiles need resizing up to the backbone's 224x224
pretraining resolution.

```python
class EuroSATSequence(keras.utils.PyDataset):
    def __init__(self, dataset, batch_size=16, image_size=IMAGE_SIZE, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        images, labels = [], []
        for i in batch_indices:
            sample = self.dataset[int(i)]
            image = keras.ops.convert_to_numpy(sample["image"]).astype("float32") / 3000.0
            image = keras.ops.convert_to_numpy(
                keras.ops.image.resize(image, (self.image_size, self.image_size))
            )
            images.append(image)
            labels.append(int(keras.ops.convert_to_numpy(sample["label"])))
        return np.stack(images), np.array(labels)


train_ds = EuroSAT100(root="data/eurosat100", split="train", download=True)
val_ds = EuroSAT100(root="data/eurosat100", split="val", download=True)
print(f"{len(train_ds)} train / {len(val_ds)} val images")

train_seq = EuroSATSequence(train_ds, batch_size=16, shuffle=True)
val_seq = EuroSATSequence(val_ds, batch_size=16, shuffle=False)
```

## Phase 1 — train the head only

The backbone is frozen (see the previous step), so only the new `Dense`
head's weights are updated. This is fast and gives the randomly-initialized
head a reasonable starting point before we touch the pretrained weights.

```python
finetune_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

finetune_model.fit(train_seq, validation_data=val_seq, epochs=3)
```

## Phase 2 — unfreeze and finetune end-to-end

Now unfreeze the backbone and continue training the whole model at a much
lower learning rate, so the pretrained Sentinel-2 features get gently
adapted to EuroSAT rather than overwritten.

```python
backbone.trainable = True

finetune_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

finetune_model.fit(train_seq, validation_data=val_seq, epochs=3)
```

## Save the finetuned weights

`EuroSAT100` is a 100-image tutorial subset, so treat the accuracy here as
a smoke test of the pipeline, not a real benchmark — swap in the full
`EuroSAT` dataset (or your own downstream dataset with a matching band
count) for real results.

```python
finetune_model.save_weights("ssl4eo_eurosat_finetuned.weights.h5")
print("saved ssl4eo_eurosat_finetuned.weights.h5")
```

## Next steps

- Swap `EuroSAT100` for the full `EuroSAT`, or any other classification
  dataset in [`keras_climate.datasets`](../api/datasets.md).
- Swap `ssl4eo_resnet50_moco` for any other encoder-only pretrained model
  in [Pretrained Weights](../pretrained-weights.md) — the freeze →
  unfreeze pattern above applies unchanged.
- For pixel-wise (segmentation) finetuning instead of classification,
  attach a decoder head to the backbone's feature map instead of
  `GlobalAveragePooling2D` + `Dense` — see `DeepLabV3Plus` in
  [`keras_climate.remote_sensing`](../api/remote-sensing.md) for a worked
  example of a decoder built on top of a ResNet backbone.
