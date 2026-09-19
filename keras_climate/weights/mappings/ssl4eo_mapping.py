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
