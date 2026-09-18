"""
Mapping for `keras_climate.foundation.anysat.AnySatEncoder`.

Important caveat: the *officially released* AnySat checkpoint
(g-astruc/AnySat) uses a substantially more complex architecture than
`AnySatEncoder` here - a zoo of modality-specific projectors (time-series
transformers for Sentinel-1/2, plain convs for aerial/SPOT/NAIP/Planet,
special-cased "mono"/"modis" handling), patch dropout, and a final
cross-modal relative-position-encoding block - and is config-driven (which
modalities/projectors exist depends on how a specific checkpoint was
trained). Faithfully reproducing that is out of scope here.

`AnySatEncoder` is instead this repo's own self-consistent "any modality,
any resolution" ViT: per-modality Conv2D patch embedding + a continuous
sinusoidal position encoding of each patch's real-world (x, y) center
(scaled by GSD), rather than the official implementation's exact
mechanism. This mapper therefore targets *that* design, for porting
weights from a from-scratch model trained with this exact architecture -
not the official public checkpoint.
"""

import re

from keras_climate.weights.mappings.vit_mapping import build_vit_mapper


def build_anysat_mapper(modalities, keras_prefix="anysat_encoder"):
    """torch_key -> keras_key mapper, assuming a source checkpoint using
    this repo's own naming convention (mirroring the Keras structure
    one-to-one): `cls_token`, `modtok_{modality}`, `embed_{modality}.proj.
    {weight,bias}`, `pos_encoding.coord_mlp.{fc1,fc2}.{weight,bias}`,
    `blocks.{i}...` (standard ViT block, handled by `build_vit_mapper`),
    `norm.{weight,bias}`."""
    rules = [
        (r"^cls_token$", f"{keras_prefix}/cls_token"),
        (r"^pos_encoding\.coord_mlp\.fc1\.weight$", f"{keras_prefix}/pos_encoding/coord_mlp/fc1/kernel"),
        (r"^pos_encoding\.coord_mlp\.fc1\.bias$", f"{keras_prefix}/pos_encoding/coord_mlp/fc1/bias"),
        (r"^pos_encoding\.coord_mlp\.fc2\.weight$", f"{keras_prefix}/pos_encoding/coord_mlp/fc2/kernel"),
        (r"^pos_encoding\.coord_mlp\.fc2\.bias$", f"{keras_prefix}/pos_encoding/coord_mlp/fc2/bias"),
    ]
    for m in modalities:
        rules += [
            (rf"^modtok_{re.escape(m)}$", f"{keras_prefix}/modtok_{m}"),
            (rf"^embed_{re.escape(m)}\.proj\.weight$", f"{keras_prefix}/embed_{m}/proj/kernel"),
            (rf"^embed_{re.escape(m)}\.proj\.bias$", f"{keras_prefix}/embed_{m}/proj/bias"),
        ]

    return build_vit_mapper(keras_prefix, block_name_fn=lambda i: f"block{i}", extra_rules=rules)
