# Examples (Colab notebooks)

Runnable Jupyter notebooks, in `examples/` in the repo — each one opens
directly in Google Colab (no local setup needed) via the badge below, and
each notebook's first cell installs `keras_climate` itself.

## [01 · Quickstart](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/01_quickstart.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/01_quickstart.ipynb)

Install `keras_climate`, pick a Keras 3 backend (TensorFlow/JAX/PyTorch),
build a segmentation model (`UNet`) and a forecasting model (`PatchTST`)
from scratch, and train with the standard `model.fit()` API.

## [02 · Run inference with a pretrained model](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/02_pretrained_inference.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/02_pretrained_inference.ipynb)

Load `unet_carvana` — a real, publicly hosted checkpoint, downloaded and
converted automatically — and run it on a photo you upload, no training
required.

## [03 · Finetune a pretrained model](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/03_finetune_pretrained_model.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/03_finetune_pretrained_model.ipynb)

Load `ssl4eo_resnet50_moco` (a ResNet-50 backbone self-supervised
pretrained on 13-band Sentinel-2 imagery), attach a new classification
head, and finetune it on `EuroSAT100` — freezing the backbone for a first
pass, then unfreezing and finetuning end-to-end at a lower learning rate.
This pattern generalizes to any encoder-only pretrained model in
[Pretrained Weights](pretrained-weights.md).

---

Prefer to run locally instead of on Colab? Each notebook works unchanged in
any Jupyter environment — just skip the `!pip install` cell if you already
have `keras_climate` installed, and the `google.colab.files.upload()` step
in notebook 02 falls back to a placeholder image automatically outside
Colab.
