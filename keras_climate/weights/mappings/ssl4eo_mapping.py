"""
Mapping for `keras_climate.remote_sensing.ssl4eo.SSL4EOResNet50`. The
backbone is `resnet_backbone(..., output_stride=32)` from
`deeplabv3plus.py`, which follows the same torchvision-standard
`conv1`/`bn1`/`layer{1..4}.{i}...` state_dict naming as DeepLabV3+'s
backbone - so `build_resnet_backbone_mapper` (shared with that module)
already covers every rule; this just wraps it as a standalone
`torch_key -> keras_key` callable with `keras_prefix="backbone"` to match
`SSL4EOResNet50`'s naming.
"""

import re

from keras_climate.weights.mappings.deeplabv3plus_mapping import build_resnet_backbone_mapper


def build_ssl4eo_mapper(keras_prefix="backbone", layer_counts=(3, 4, 6, 3)):
    rules = build_resnet_backbone_mapper(keras_prefix, layer_counts)
    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
