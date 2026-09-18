"""
Mapping for `keras_climate.weather.metnet.MetNet`.

Important caveat: the original MetNet/MetNet-2 (Sønderby et al. 2020;
Espeholt et al. 2022) is a Google Research model with no public code or
weights release. Community reimplementations exist (e.g.
`openclimatefix/metnet`, which does have a small downloadable checkpoint),
but they use a substantially different set of building blocks - a
space-to-depth preprocessor, a `ConvGRU` (not `ConvLSTM`) temporal encoder,
and attention/positional-embedding mechanics from the external
`axial_attention` package - that don't line up with this repo's simplified
design (a dilated-conv context tower + a from-scratch axial self-attention
layer capturing MetNet's core "large context + spatial aggregation" idea).
This mapper therefore targets *this* architecture, for porting weights
from a from-scratch model trained with it - not a real MetNet checkpoint.

The temporal encoder is a plain Keras `ConvLSTM2D`, so its conversion
reuses `convert_convlstm_stack_state_dict` (see `convlstm_mapping.py` for
why that needs gate-reordering/kernel-splitting, not just a transpose).
"""

import re

import numpy as np

from keras_climate.weights.mappings.convlstm_mapping import convert_convlstm_stack_state_dict


def convert_metnet_state_dict(flat_state_dict, base_filters, num_dilated_convs=6,
                               num_att_layers=4, num_upsample=2):
    """Translates a source state_dict using this repo's own naming
    convention into `{keras_weight_path: np.ndarray}`.

    Note: the temporal ConvLSTM's *input* channel count is `base_filters`,
    not the raw frame channel count - it consumes the stem conv's output
    feature map, not the original frames.

    Assumed source naming (mirrors the Keras structure 1:1):
        stem.conv.{weight,bias}, stem_bn.bn.{weight,bias,running_mean,running_var}
        temporal_encoder.cell_list.0.conv.{weight,bias}   (ConvLSTM cell, ndrplz-style fused conv)
        context_tower.{i}.conv.{weight,bias}, context_tower.{i}.bn.{...}
        lead_time_embed.weight                            (nn.Embedding)
        attn_proj_in.{weight,bias}
        axial_attn{i}.{row,col}_{qkv,proj}.{weight,bias}, axial_attn{i}.norm{1,2}.{weight,bias}
        upsample{i}.{weight,bias}, upsample_bn{i}.{...}
        precip_head.{weight,bias}
    """
    out = {}

    def copy(torch_key, keras_key):
        if torch_key in flat_state_dict:
            out[keras_key] = flat_state_dict[torch_key]

    copy("stem.conv.weight", "stem/conv/kernel")
    copy("stem.conv.bias", "stem/conv/bias")
    copy("stem_bn.bn.weight", "stem_bn/bn/gamma")
    copy("stem_bn.bn.bias", "stem_bn/bn/beta")
    copy("stem_bn.bn.running_mean", "stem_bn/bn/moving_mean")
    copy("stem_bn.bn.running_var", "stem_bn/bn/moving_variance")

    temporal_flat = {k[len("temporal_encoder."):]: v for k, v in flat_state_dict.items()
                      if k.startswith("temporal_encoder.")}
    if temporal_flat:
        temporal_converted = convert_convlstm_stack_state_dict(
            temporal_flat, [(base_filters, base_filters)], keras_prefix="temporal_encoder",
        )
        # single-layer stack: rename "temporal_encoder0" -> "temporal_encoder"
        for k, v in temporal_converted.items():
            out[k.replace("temporal_encoder0", "temporal_encoder", 1)] = v

    for i in range(num_dilated_convs):
        copy(f"context_tower.{i}.conv.weight", f"context_tower_dconv{i}/kernel")
        copy(f"context_tower.{i}.conv.bias", f"context_tower_dconv{i}/bias")
        copy(f"context_tower.{i}.bn.weight", f"context_tower_bn{i}/gamma")
        copy(f"context_tower.{i}.bn.bias", f"context_tower_bn{i}/beta")
        copy(f"context_tower.{i}.bn.running_mean", f"context_tower_bn{i}/moving_mean")
        copy(f"context_tower.{i}.bn.running_var", f"context_tower_bn{i}/moving_variance")

    copy("lead_time_embed.weight", "lead_time_embed/embeddings")
    copy("attn_proj_in.weight", "attn_proj_in/kernel")
    copy("attn_proj_in.bias", "attn_proj_in/bias")

    for i in range(num_att_layers):
        p, kp = f"axial_attn{i}", f"axial_attn{i}"
        for stream in ("row", "col"):
            copy(f"{p}.{stream}_qkv.weight", f"{kp}/{stream}_qkv/kernel")
            copy(f"{p}.{stream}_qkv.bias", f"{kp}/{stream}_qkv/bias")
            copy(f"{p}.{stream}_proj.weight", f"{kp}/{stream}_proj/kernel")
            copy(f"{p}.{stream}_proj.bias", f"{kp}/{stream}_proj/bias")
        for n in ("norm1", "norm2"):
            copy(f"{p}.{n}.weight", f"{kp}/{n}/gamma")
            copy(f"{p}.{n}.bias", f"{kp}/{n}/beta")

    for i in range(num_upsample):
        copy(f"upsample{i}.weight", f"upsample{i}/kernel")
        copy(f"upsample{i}.bias", f"upsample{i}/bias")
        copy(f"upsample_bn{i}.weight", f"upsample_bn{i}/gamma")
        copy(f"upsample_bn{i}.bias", f"upsample_bn{i}/beta")
        copy(f"upsample_bn{i}.running_mean", f"upsample_bn{i}/moving_mean")
        copy(f"upsample_bn{i}.running_var", f"upsample_bn{i}/moving_variance")

    copy("precip_head.weight", "precip_head/kernel")
    copy("precip_head.bias", "precip_head/bias")

    return out


def build_metnet_mapper():
    """`convert_metnet_state_dict` already produces keys spelled exactly as
    the target Keras weight paths."""
    return lambda k: k
