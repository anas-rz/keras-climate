"""
keras_climate.remote_sensing.deeplabv3plus
--------------------------------------------
DeepLabV3+ (Chen et al. 2018): dilated-ResNet backbone + ASPP + light
decoder with a low-level-feature skip connection. Commonly used for
land-cover classification and semantic segmentation of aerial/satellite
imagery.
"""

import keras
from keras import layers, ops
from keras_climate.utils.layers import ConvBNAct, ASPP


def _bottleneck_block(x, filters, stride=1, dilation=1, downsample=False, name=""):
    shortcut = x
    if downsample:
        shortcut = layers.Conv2D(filters * 4, 1, strides=stride, use_bias=False,
                                  name=f"{name}_downsample_conv")(x)
        shortcut = layers.BatchNormalization(epsilon=1e-5, name=f"{name}_downsample_bn")(shortcut)

    x = ConvBNAct(filters, 1, name=f"{name}_conv1")(x)
    x = ConvBNAct(filters, 3, strides=stride, dilation_rate=dilation, name=f"{name}_conv2")(x)
    x = layers.Conv2D(filters * 4, 1, use_bias=False, name=f"{name}_conv3")(x)
    x = layers.BatchNormalization(epsilon=1e-5, name=f"{name}_bn3")(x)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_out")(x)
    return x


def _resnet_stage(x, filters, num_blocks, stride, dilation, name):
    x = _bottleneck_block(x, filters, stride=stride, dilation=dilation,
                           downsample=True, name=f"{name}_block0")
    for i in range(1, num_blocks):
        x = _bottleneck_block(x, filters, stride=1, dilation=dilation,
                               downsample=False, name=f"{name}_block{i}")
    return x


def resnet_backbone(x, layer_counts=(3, 4, 23, 3), output_stride=16, name="resnet"):
    """A ResNet-50/101-style backbone with atrous convolutions in the last
    stage(s) so the overall output stride matches DeepLab's requirements
    (8 or 16), returning (low_level_feat, high_level_feat)."""
    if output_stride == 16:
        strides = [1, 2, 2, 1]
        dilations = [1, 1, 1, 2]
    elif output_stride == 8:
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]
    else:
        raise ValueError("output_stride must be 8 or 16")

    x = ConvBNAct(64, 7, strides=2, name=f"{name}_stem")(x)
    # padding="same" would pad asymmetrically here (see ConvBNAct); the
    # reference torchvision resnet stem uses `nn.MaxPool2d(3, stride=2,
    # padding=1)`, a symmetric pad of 1 on each side.
    x = layers.ZeroPadding2D(1, name=f"{name}_stem_pool_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="valid", name=f"{name}_stem_pool")(x)

    x = _resnet_stage(x, 64, layer_counts[0], strides[0], dilations[0], f"{name}_stage1")
    low_level_feat = x  # stride-4 features for the decoder skip connection

    x = _resnet_stage(x, 128, layer_counts[1], strides[1], dilations[1], f"{name}_stage2")
    x = _resnet_stage(x, 256, layer_counts[2], strides[2], dilations[2], f"{name}_stage3")
    x = _resnet_stage(x, 512, layer_counts[3], strides[3], dilations[3], f"{name}_stage4")

    return low_level_feat, x


def DeepLabV3Plus(
    input_shape=(512, 512, 3),
    num_classes=1,
    backbone_layers=(3, 4, 23, 3),  # ResNet-101; use (3,4,6,3) for ResNet-50
    output_stride=16,
    aspp_filters=256,
    decoder_filters=48,
    final_activation=None,
    name="deeplabv3plus",
):
    inputs = keras.Input(shape=input_shape, name="image")

    low_level_feat, x = resnet_backbone(inputs, backbone_layers, output_stride, name="backbone")

    x = ASPP(filters=aspp_filters, name="aspp")(x)

    x = layers.Lambda(lambda t: ops.image.resize(t[0], ops.shape(t[1])[1:3], interpolation="bilinear"),
                       name="upsample_to_low_level")([x, low_level_feat])

    low_level_feat = ConvBNAct(decoder_filters, 1, name="low_level_project")(low_level_feat)

    x = layers.Concatenate(name="decoder_concat")([x, low_level_feat])
    x = ConvBNAct(256, 3, name="decoder_conv1")(x)
    x = ConvBNAct(256, 3, name="decoder_conv2")(x)

    x = layers.Lambda(
        lambda t: ops.image.resize(t, (input_shape[0], input_shape[1]), interpolation="bilinear"),
        name="upsample_to_input",
    )(x)

    outputs = layers.Conv2D(num_classes, 1, activation=final_activation, name="logits")(x)

    return keras.Model(inputs, outputs, name=name)


def deeplabv3plus_config(variant="resnet50"):
    presets = {
        "resnet50": dict(backbone_layers=(3, 4, 6, 3)),
        "resnet101": dict(backbone_layers=(3, 4, 23, 3)),
    }
    return presets[variant]
