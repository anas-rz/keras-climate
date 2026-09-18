# keras_climate

A Keras 3 (TensorFlow / JAX / PyTorch backend) framework of models for
climate modeling and Earth observation, plus tooling to port pretrained
weights from the original reference implementations (almost always
PyTorch) onto the Keras equivalents.

## Structure

```
keras_climate
├── remote_sensing        pixel-wise segmentation / classification of imagery
│   ├── unet               classic encoder-decoder segmentation
│   ├── deeplabv3plus       dilated-ResNet + ASPP segmentation
│   ├── segformer           MiT transformer backbone + all-MLP decode head
│   └── satmae              ViT masked-autoencoder for multispectral/temporal imagery
│
├── forecasting            multivariate time-series forecasting
│   ├── patchtst            channel-independent patch transformer
│   ├── timesnet             FFT period discovery + Inception 2D blocks
│   └── tft                  Temporal Fusion Transformer (LSTM + attention)
│
├── weather                spatiotemporal nowcasting / weather models
│   ├── convlstm             stacked ConvLSTM encoder-forecaster
│   ├── earthformer          hierarchical cuboid-attention transformer
│   └── metnet               dilated-conv context tower + axial attention
│
├── foundation             large pretrained Earth-observation foundation models
│   ├── prithvi              spatiotemporal ViT-MAE for NASA HLS imagery
│   ├── clay                 sensor-agnostic ViT with band/geo metadata embeddings
│   ├── croma                dual SAR+optical encoder with cross-attention fusion
│   └── anysat                any-resolution / any-modality ViT
│
├── utils/layers.py         shared building blocks (attention, patch embed, conv blocks)
│
└── weights/                generic PyTorch -> Keras weight-porting framework
    ├── converter.py          WeightConverter, transpose inference, checkpoint loaders
    ├── mappings/              example / reusable name-mapping rule sets
    └── port_weights.py        CLI entry point + per-model registry
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

## Porting pretrained weights

The weight-porting framework in `keras_climate.weights` is *generic*: you
describe how PyTorch checkpoint keys map to Keras weight names (either as
regex rules or a Python callable), and `WeightConverter` handles shape
inference (Dense/Conv kernel transposition), assignment, and reports any
weights it could not match on either side.

```python
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

### Adding a new mapping

1. Get the source `state_dict` key names: `list(state_dict.keys())`.
2. Get the target Keras weight names: `[w.path for w in model.weights]`.
3. Write regex rules (see `weights/mappings/unet_mapping.py` for a
   hand-written example, or `weights/mappings/vit_mapping.py` for a
   reusable generic-ViT-block mapper used by every transformer model in
   this repo).
4. Run `converter.convert(strict=False)` first and inspect the "no source
   match" / "unused source keys" report to iterate on the rules; switch to
   `strict=True` once the report is clean.
5. Register the (build_fn, name_map_fn, loader) triple in
   `MODEL_REGISTRY` inside `port_weights.py` so it's usable from the CLI.

Note: exact checkpoint key names vary by which reference repo/checkpoint
you're porting from (e.g. the official author repo vs. a HuggingFace
re-upload vs. an `mmsegmentation` config) - the mappings shipped here are
worked examples of the *pattern*, not guaranteed to match every public
checkpoint byte-for-byte without adjustment.
