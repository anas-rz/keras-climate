# keras_climate

A Keras 3 (TensorFlow / JAX / PyTorch backend) framework of models for
climate modeling and Earth observation, plus tooling to port pretrained
weights from the original reference implementations (almost always
PyTorch) onto the Keras equivalents.

**[Full documentation](https://anas-rz.github.io/keras-climate/)**

## Install

```bash
pip install -e .
```

## Quick start

```python
from keras_climate.remote_sensing import UNet, SegFormer
from keras_climate.forecasting import PatchTST
from keras_climate.weather import ConvLSTMNowcaster
from keras_climate.foundation import PrithviClassifier

model = UNet(input_shape=(256, 256, 4), num_classes=5)   # 4-band (e.g. RGB+NIR) input
model.summary()

forecaster = PatchTST(seq_len=336, pred_len=96, num_channels=7)
```

All builders return a plain `keras.Model`, so standard `model.compile(...)`
/ `model.fit(...)` / `model.save(...)` all work unchanged.

See the [documentation](https://anas-rz.github.io/keras-climate/) for the
full model roster, the pretrained-weight-porting guide, and which models
ship a ready-to-download real checkpoint.
