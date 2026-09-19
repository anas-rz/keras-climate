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
├── utils/layers.py         shared building blocks (attention, patch embed, conv blocks)
└── weights/                generic PyTorch -> Keras weight-porting framework
    ├── converter.py          WeightConverter, transpose inference, checkpoint loaders
    ├── mappings/              example / reusable name-mapping rule sets
    ├── pretrained.py          timm-style loaders for real, publicly-hosted checkpoints
    └── port_weights.py        CLI entry point + per-model registry
```

See the [Models](models/remote-sensing.md) section for what each family
contains, [Weight Porting](weight-porting.md) for how the conversion
framework works, and [Pretrained Weights](pretrained-weights.md) for which
models ship a ready-to-download real checkpoint.

## Installation

```bash
pip install -e .
```

Then install a Keras 3 backend of your choice:

```bash
pip install tensorflow      # or
pip install "jax[cpu]"      # or
pip install torch
```

See [Quick Start](quickstart.md) for a minimal end-to-end example.
