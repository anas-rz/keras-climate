# keras_climate

A Keras 3 (TensorFlow / JAX / PyTorch backend) framework of models for
climate modeling and Earth observation, plus tooling to port pretrained
weights from the original reference implementations (almost always
PyTorch) onto the Keras equivalents.

Every model is a plain `keras.Model` (or `keras.Model` subclass) builder
function — no custom training loop is imposed, so they compose with
standard `model.fit(...)`, custom losses, distribution strategies, etc.

## Package structure

```
keras_climate
├── remote_sensing        pixel-wise segmentation / classification of imagery
├── forecasting            multivariate time-series forecasting
├── weather                spatiotemporal nowcasting / weather models
├── foundation             large pretrained Earth-observation foundation models
├── operators              neural operator learning (PDE surrogate modeling)
├── datasets/              Keras port of torchgeo.datasets (190+ concrete datasets)
├── samplers/               Keras port of torchgeo.samplers (geospatial tile sampling)
├── transforms/             Keras port of torchgeo.transforms (band indices, SAR, ...)
├── losses/                 Keras port of torchgeo.losses (ELECTS, QR/RQ)
├── utils/layers.py         shared building blocks (attention, patch embed, conv blocks)
└── weights/                generic PyTorch -> Keras weight-porting framework
    ├── converter.py          WeightConverter, transpose inference, checkpoint loaders
    ├── mappings/              example / reusable name-mapping rule sets
    ├── pretrained.py          timm-style loaders for real, publicly-hosted checkpoints
    └── port_weights.py        CLI entry point + per-model registry
```

See the [Models](models/remote-sensing.md) section for what each family
contains, [Data](models/data.md) for the dataset/sampler/transform
framework, [Weight Porting](weight-porting.md) for how the conversion
framework works, and [Pretrained Weights](pretrained-weights.md) for which
models ship a ready-to-download real checkpoint.

## Runnable examples

Every example below opens directly in Google Colab, no local setup needed
— see [Examples](examples.md) for the full list, including a complete
**finetune a pretrained model** walkthrough.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/keras-climate/blob/main/examples/01_quickstart.ipynb)

## Installation

```bash
pip install git+https://github.com/anas-rz/keras-climate
```

Then install a Keras 3 backend of your choice:

```bash
pip install tensorflow      # or
pip install "jax[cpu]"      # or
pip install torch
```

See [Quick Start](quickstart.md) for a minimal end-to-end example.
