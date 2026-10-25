# Build and train models from scratch

**Backend:** Multi-Backend (TensorFlow / JAX / PyTorch)
**Models:** `UNet`, `PatchTST`
**Description:** Install `keras_climate`, pick a Keras 3 backend, build a
segmentation model and a forecasting model from scratch, and train with
the standard `model.fit()` API.

[View in Colab](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/01_quickstart.ipynb){ .md-button }
[GitHub source](https://github.com/anas-rz/keras-climate/blob/main/examples/01_quickstart.ipynb){ .md-button }

---

`keras_climate` is a [Keras 3](https://keras.io/keras_3/) (TensorFlow / JAX
/ PyTorch backend) framework of models for climate modeling and Earth
observation, plus tooling to port pretrained weights from the original
reference implementations (almost always PyTorch) onto the Keras
equivalents.

This walkthrough covers the basics: installing `keras_climate` and picking
a backend, building a segmentation model and a forecasting model from
scratch, and training with the standard `model.fit()` API.

## Install

```bash
pip install -q keras-climate
```

## Pick a Keras 3 backend

Colab ships TensorFlow by default, which Keras 3 uses automatically. You
can switch to JAX or PyTorch instead by setting `KERAS_BACKEND` **before**
`keras` is imported anywhere in the process.

```python
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # or "jax" / "torch"

import keras

print("Keras version:", keras.__version__)
print("Backend:", keras.backend.backend())
```

## Build a segmentation model

Every model in `keras_climate` is a plain `keras.Model` builder function —
no custom training loop is imposed, so `model.compile()` / `model.fit()` /
`model.save()` all work exactly as they would for any other Keras model.

```python
from keras_climate.remote_sensing import UNet

# 4-band input (e.g. RGB + near-infrared), 5 land-cover classes
model = UNet(input_shape=(256, 256, 4), num_classes=5)
model.summary()
```

## Build a forecasting model

```python
from keras_climate.forecasting import PatchTST
import numpy as np

forecaster = PatchTST(seq_len=336, pred_len=96, num_channels=7)

x = np.random.randn(2, 336, 7).astype("float32")
y = forecaster(x)
print("forecast shape:", y.shape)  # (batch, pred_len, num_channels)
```

## Train like any other Keras model

This uses random data purely to demonstrate the plumbing — swap in a real
`keras_climate.datasets` loader (see the [Data](../models/data.md) guide)
or your own `tf.data`/NumPy pipeline for real training.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

x = np.random.randn(4, 256, 256, 4).astype("float32")
y = np.random.randint(0, 5, size=(4, 256, 256)).astype("int32")

model.fit(x, y, epochs=1, batch_size=2)
```

## Next steps

- **[Run inference with a pretrained UNet](pretrained-inference.md)** —
  several models ship a loader that downloads a real, publicly hosted
  checkpoint and returns a model with pretrained weights already loaded.
- **[Finetune a pretrained Sentinel-2 backbone on EuroSAT](finetune-pretrained-model.md)**.
