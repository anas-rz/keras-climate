"""
Mapping for `keras_climate.remote_sensing.scalemae.ScaleMAEEncoder`. The
encoder is a standard timm/MAE ViT-Large otherwise, so `build_vit_mapper`
(shared with SatMAE/Prithvi/AnySat) already covers every block/patch-embed/
norm/cls_token rule - the one ScaleMAE-specific wrinkle is that the source
checkpoint's `pos_embed` buffer is dead weight (never added in the official
`forward_encoder` - ScaleMAE recomputes a GSD-aware positional embedding
live from each sample's ground-sample-distance instead of loading a fixed
table, see `GSDPositionalEmbedding`) and must be skipped rather than mapped
onto anything.
"""

from keras_climate.weights.mappings.vit_mapping import build_vit_mapper


def build_scalemae_mapper(keras_prefix="scalemae_encoder"):
    return build_vit_mapper(keras_prefix)


SCALEMAE_SKIP_PATTERNS = [r"^pos_embed$", r"^decoder_pos_embed$"]
