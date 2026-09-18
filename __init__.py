"""
keras_climate
=============
A Keras 3 (TF / JAX / Torch backend) framework of models for Earth
observation and climate modeling, organized into four families:

    keras_climate.remote_sensing  - UNet, DeepLabV3+, SegFormer, SatMAE
    keras_climate.forecasting     - PatchTST, TimesNet, TFT
    keras_climate.weather         - ConvLSTM, Earthformer, MetNet
    keras_climate.foundation      - Prithvi, Clay, CROMA, AnySat

Each model is a plain `keras.Model` (or `keras.Model` subclass) builder
function - no custom training loop is imposed, so they compose with
standard `model.fit(...)`, custom losses, distribution strategies, etc.

Weight porting from the original (usually PyTorch) reference
implementations is handled by `keras_climate.weights`; see
`keras_climate/weights/README.md` for the conversion workflow.
"""

__version__ = "0.1.0"

from . import remote_sensing
from . import forecasting
from . import weather
from . import foundation
from . import weights

__all__ = ["remote_sensing", "forecasting", "weather", "foundation", "weights", "__version__"]
