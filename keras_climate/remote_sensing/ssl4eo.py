"""
keras_climate.remote_sensing.ssl4eo
---------------------------------------
SSL4EO-S12 (Wang et al. 2022): not a novel architecture but a large-scale
Sentinel-1/Sentinel-2 dataset paired with several self-supervised (MoCo v2,
DINO, MAE, data2vec) pretrained backbones. This module ports the released
ResNet-50 backbone - a standard torchvision ResNet-50 with its stem's first
conv widened from 3 to 13 input channels (Sentinel-2 L1C's full band set)
and otherwise architecturally unchanged - reusing `resnet_backbone` from
`keras_climate.remote_sensing.deeplabv3plus` with the plain (non-dilated,
output_stride=32) stride pattern rather than DeepLab's atrous one.
"""

import keras
from keras_climate.remote_sensing.deeplabv3plus import resnet_backbone


def SSL4EOResNet50(input_shape=(224, 224, 13), name="ssl4eo_resnet50"):
    """Feature-extractor model: image -> final-stage feature map
    `(B, H/32, W/32, 2048)`, for attaching a downstream head."""
    inputs = keras.Input(shape=input_shape, name="image")
    _, features = resnet_backbone(inputs, layer_counts=(3, 4, 6, 3), output_stride=32,
                                   name="backbone")
    return keras.Model(inputs, features, name=name)
