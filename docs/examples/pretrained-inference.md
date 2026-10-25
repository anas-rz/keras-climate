# Run inference with a pretrained UNet

**Backend:** Multi-Backend (TensorFlow / JAX / PyTorch)
**Model:** `unet_carvana` (`UNet`)
**Dataset/Checkpoint:** milesial/Pytorch-UNet's Carvana release
**Description:** Load a real, publicly hosted checkpoint and run it on a
photo you upload — no training required.

[View in Colab](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/02_pretrained_inference.ipynb){ .md-button }
[GitHub source](https://github.com/anas-rz/keras-climate/blob/main/examples/02_pretrained_inference.ipynb){ .md-button }

---

`keras_climate.weights.pretrained` provides "timm-style" pretrained-weight
loaders: each function downloads a real, publicly hosted PyTorch checkpoint
(cached locally so it's only fetched once), ports it through this repo's
`WeightConverter` + the matching name-mapping, and returns a ready
`keras.Model` with real pretrained weights loaded — **no training
required** to get a working model.

This walkthrough loads `unet_carvana` — milesial/Pytorch-UNet's own
released weights for binary car/background segmentation — and runs it on a
photo you upload (or a synthetic placeholder if you're not running this in
Colab).

See the full coverage table in
[Pretrained Weights](../pretrained-weights.md) for every model with a real,
loadable checkpoint (segmentation, foundation models, weather models, ...).

## Install

```bash
pip install -q keras-climate
```

```python
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
import matplotlib.pyplot as plt
```

## Load the pretrained model

```python
from keras_climate.weights.pretrained import unet_carvana

model, report = unet_carvana()

n_matched = len(report["matched"])
n_total = n_matched + len(report["missing_in_source"])
print(f"Converted {n_matched}/{n_total} weights from the real checkpoint.")
print(f"Unused source keys: {len(report['unused_source_keys'])}")
```

## Get an input image

If this notebook is running in Colab, you'll be prompted to upload a photo
(ideally a car on a plain-ish background, matching what the model was
trained on). Otherwise — or if you skip the upload — a random placeholder
image is used instead, purely so the rest of the notebook still runs.

```python
image_path = None

try:
    from google.colab import files

    print("Upload a photo (JPG/PNG) — ideally a car:")
    uploaded = files.upload()
    if uploaded:
        image_path = next(iter(uploaded))
except ImportError:
    pass

input_size = (256, 256)  # unet_carvana()'s default input_shape=(256, 256, 3)

if image_path:
    img = keras.utils.load_img(image_path, target_size=input_size)
    image = keras.utils.img_to_array(img) / 255.0
else:
    print("No upload available — using a random placeholder image instead.")
    image = np.random.rand(*input_size, 3).astype("float32")

x = image[None, ...].astype("float32")
print("input shape:", x.shape)
```

## Run inference

```python
logits = model.predict(x)
mask = np.argmax(logits[0], axis=-1)  # (H, W), values in {0: background, 1: car}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image)
axes[0].set_title("input")
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("predicted mask")
axes[2].imshow(image)
axes[2].imshow(mask, cmap="jet", alpha=0.4)
axes[2].set_title("overlay")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## Other pretrained models

Every loader follows the same `model, report = ...()` pattern. A few
examples (not run here — they need domain-specific inputs, e.g. multi-band
Sentinel-2 imagery):

```python
from keras_climate.weights.pretrained import croma_base, prithvi_eo_100m, climax_1_40625deg

croma_model, report = croma_base()             # SAR + optical foundation model
prithvi_model, report = prithvi_eo_100m()       # NASA/IBM HLS foundation model (encoder)
climax_model, report = climax_1_40625deg()      # global weather forecasting
```

See [Pretrained Weights](../pretrained-weights.md) for the full coverage
table, and
**[Finetune a pretrained Sentinel-2 backbone on EuroSAT](finetune-pretrained-model.md)**
for how to adapt one of these to your own downstream task.
