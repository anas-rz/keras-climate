import re

from keras_climate.weights.mappings.vit_mapping import build_vit_mapper


def build_anysat_mapper(modalities, keras_prefix="anysat_encoder"):
    rules = [
        (r"^cls_token$", f"{keras_prefix}/cls_token"),
        (
            r"^pos_encoding\.coord_mlp\.fc1\.weight$",
            f"{keras_prefix}/pos_encoding/coord_mlp/fc1/kernel",
        ),
        (
            r"^pos_encoding\.coord_mlp\.fc1\.bias$",
            f"{keras_prefix}/pos_encoding/coord_mlp/fc1/bias",
        ),
        (
            r"^pos_encoding\.coord_mlp\.fc2\.weight$",
            f"{keras_prefix}/pos_encoding/coord_mlp/fc2/kernel",
        ),
        (
            r"^pos_encoding\.coord_mlp\.fc2\.bias$",
            f"{keras_prefix}/pos_encoding/coord_mlp/fc2/bias",
        ),
    ]
    for m in modalities:
        rules += [
            (rf"^modtok_{re.escape(m)}$", f"{keras_prefix}/modtok_{m}"),
            (
                rf"^embed_{re.escape(m)}\.proj\.weight$",
                f"{keras_prefix}/embed_{m}/proj/kernel",
            ),
            (
                rf"^embed_{re.escape(m)}\.proj\.bias$",
                f"{keras_prefix}/embed_{m}/proj/bias",
            ),
        ]

    return build_vit_mapper(
        keras_prefix, block_name_fn=lambda i: f"block{i}", extra_rules=rules
    )
