# Quick Start

## Building a model

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

## Loading real pretrained weights

Several models ship a ready-made loader that downloads a real, publicly
hosted checkpoint and returns a model with pretrained weights already
loaded — see [Pretrained Weights](pretrained-weights.md) for full coverage:

```python
from keras_climate.weights.pretrained import unet_carvana

model, report = unet_carvana()
model.summary()
```

## Porting your own checkpoint

For models without a built-in loader, use the generic weight-porting
framework directly — see [Weight Porting](weight-porting.md) for the full
guide:

```python
import keras
from keras_climate.remote_sensing import UNet
from keras_climate.weights import WeightConverter, load_torch_state_dict_as_numpy
from keras_climate.weights.mappings import build_unet_mapper

model = UNet(input_shape=(256, 256, 3), num_classes=1)
model(keras.ops.zeros((1, 256, 256, 3)))  # build the model

state_dict = load_torch_state_dict_as_numpy("unet_carvana.pth")
converter = WeightConverter(model, state_dict, build_unet_mapper())
report = converter.convert(strict=False)

model.save_weights("unet_keras.weights.h5")
```

Or via the CLI wrapper, which bundles a model-build function + name-mapper
per checkpoint family in `weights/port_weights.py`:

```bash
python -m keras_climate.weights.port_weights \
    --model unet --checkpoint unet_carvana.pth --output unet_keras.weights.h5
```
