"""
keras_climate.operators.uno
-------------------------------
UNO (Rahman et al. 2022, "U-Shaped Neural Operators"): a U-Net-style
encoder-decoder where each conv block is a spectral (FNO-style) operator
instead of a local convolution, so the whole network stays resolution-
invariant while still getting U-Net's multi-scale skip-connection
structure.

Scope note: the original paper's spectral-conv blocks change spatial
resolution *directly* via the FFT (truncating to a different output grid
size on the inverse transform, avoiding a separate pooling op). This
module instead uses standard `AveragePooling2D`/`UpSampling2D` between
levels for resolution changes, keeping each level's `SpectralConv2D`
operating at a fixed resolution - a simpler, easier-to-verify structure
that keeps the essential "U-Net built from spectral instead of local
convs" idea without the added complexity of a resolution-changing
spectral transform. No pretrained checkpoint exists for UNO anywhere
(neither the paper's own code nor the `neuraloperator` library ships one)
- validated against a from-scratch synthetic PyTorch reference of this
module's own architecture (see `weights/mappings/uno_mapping.py`).
"""

import keras
from keras import layers
from keras_climate.operators.fno import SpectralConv2D


class UNOBlock(layers.Layer):
    """Spectral conv + pointwise (1x1) conv skip, summed, then GELU -
    same combination as `keras_climate.operators.fno.FNOBlock`, but
    supporting `in_channels != out_channels` (inferred from the input at
    `build()` time) since UNO changes channel width between levels."""

    def __init__(self, out_channels, modes1, modes2, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.spectral = SpectralConv2D(in_channels, self.out_channels, self.modes1, self.modes2,
                                        name="spectral")
        self.pointwise = layers.Conv2D(self.out_channels, 1, name="pointwise")
        self.act = layers.Activation("gelu")
        super().build(input_shape)

    def call(self, x):
        return self.act(self.spectral(x) + self.pointwise(x))


def UNO(input_shape=(64, 64, 1), out_channels=1, base_width=16, depth=3, modes1=8, modes2=8,
        name="uno"):
    """U-shaped neural operator: lift -> `depth` downsampling spectral
    levels -> spectral bottleneck -> `depth` upsampling spectral levels
    (with encoder skip connections) -> project. `input_shape`'s spatial
    dims must be divisible by `2**depth`, and `modes1`/`modes2` must be at
    most half the bottleneck resolution."""
    inputs = keras.Input(shape=input_shape, name="field")
    x = layers.Dense(base_width, name="lift")(inputs)

    widths = [base_width * (2 ** i) for i in range(depth + 1)]
    skips = []
    for i in range(depth):
        x = UNOBlock(widths[i], modes1, modes2, name=f"enc_block{i}")(x)
        skips.append(x)
        x = layers.AveragePooling2D(2, name=f"down{i}")(x)

    x = UNOBlock(widths[depth], modes1, modes2, name="bottleneck")(x)

    for i in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear", name=f"up{i}")(x)
        x = layers.Concatenate(name=f"concat{i}")([x, skips[i]])
        x = UNOBlock(widths[i], modes1, modes2, name=f"dec_block{i}")(x)

    out = layers.Dense(out_channels, name="project")(x)
    return keras.Model(inputs, out, name=name)
