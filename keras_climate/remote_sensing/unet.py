"""
keras_climate.remote_sensing.unet
----------------------------------
Classic UNet (Ronneberger et al. 2015) for pixel-wise segmentation of
remote-sensing rasters (land-cover, cloud masks, flood extent, etc).
Supports an arbitrary number of input bands (in_channels), not just RGB.
"""

import keras
from keras import layers
from keras_climate.utils.layers import DoubleConv


def UNet(
    input_shape=(256, 256, 3),
    num_classes=1,
    base_filters=64,
    depth=4,
    final_activation=None,
    name="unet",
):
    """Build a UNet functional model.

    Args:
        input_shape: (H, W, C) - C can be any number of spectral bands.
        num_classes: number of output classes (1 for binary/regression).
        base_filters: number of filters in the first encoder stage; doubles
            at each downsampling stage.
        depth: number of downsampling stages.
        final_activation: e.g. "sigmoid" / "softmax" / None (logits).
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs

    skips = []
    filters = base_filters
    for i in range(depth):
        x = DoubleConv(filters, name=f"enc{i}")(x)
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        filters *= 2

    x = DoubleConv(filters, name="bottleneck")(x)

    for i in reversed(range(depth)):
        filters //= 2
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                                    name=f"upconv{i}")(x)
        skip = skips[i]
        # Skip first, then the upsampled path - matches the convention used
        # by essentially every reference PyTorch UNet (e.g. `torch.cat([x2,
        # x1], dim=1)` in milesial/Pytorch-UNet), so a ported decoder conv's
        # input-channel order lines up with the source checkpoint.
        x = layers.Concatenate(name=f"concat{i}")([skip, x])
        x = DoubleConv(filters, name=f"dec{i}")(x)

    outputs = layers.Conv2D(num_classes, 1, activation=final_activation,
                             name="logits")(x)

    return keras.Model(inputs, outputs, name=name)


def unet_config(variant="base"):
    """Convenience presets matching common pretrained-checkpoint sizes."""
    presets = {
        "small": dict(base_filters=32, depth=4),
        "base": dict(base_filters=64, depth=4),
        "large": dict(base_filters=64, depth=5),
    }
    return presets[variant]
