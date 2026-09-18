"""
Mapping for `keras_climate.foundation.clay.ClayEncoder`, targeting the
official Clay-foundation/model `Encoder` naming convention.

No real Clay checkpoint is small enough to validate here (see
`foundation/clay.py`'s module docstring), so - unlike this repo's other
mappings - this one hasn't been checked against a real downloaded
checkpoint. It's built directly from the official source
(`claymodel/model.py::Encoder`, `claymodel/factory.py::DynamicEmbedding`,
`claymodel/backbone.py::Transformer`) and should be a close match, but
budget for possible small naming adjustments against whatever specific
checkpoint you're actually porting from (a real Lightning checkpoint will
also need its `model.encoder.` (or similar) prefix stripped first via
`key_prefix_strip`).
"""

import re

import numpy as np


def convert_clay_encoder_state_dict(flat_state_dict, depth=12, num_latent_tokens=128):
    """Translates a (flat, un-prefixed) Clay `Encoder` state_dict into
    `{keras_weight_path: np.ndarray}`."""
    out = {}

    def copy(torch_key, keras_key):
        if torch_key in flat_state_dict:
            out[keras_key] = flat_state_dict[torch_key]

    copy("cls_token", "clay_encoder/cls_token")

    # DynamicEmbedding
    pe = "patch_embedding"
    copy(f"{pe}.fclayer.l1.weight", "clay_encoder/patch_embedding/fclayer/l1/kernel")
    copy(f"{pe}.fclayer.l1.bias", "clay_encoder/patch_embedding/fclayer/l1/bias")
    copy(f"{pe}.fclayer.l2.weight", "clay_encoder/patch_embedding/fclayer/l2/kernel")
    copy(f"{pe}.fclayer.l2.bias", "clay_encoder/patch_embedding/fclayer/l2/bias")

    wg = f"{pe}.weight_generator"
    kwg = "clay_encoder/patch_embedding/weight_generator"
    copy(f"{wg}.weight_tokens", f"{kwg}/weight_tokens")
    copy(f"{wg}.bias_token", f"{kwg}/bias_token")
    copy(f"{wg}.fc_weight.weight", f"{kwg}/fc_weight/kernel")
    copy(f"{wg}.fc_weight.bias", f"{kwg}/fc_weight/bias")
    copy(f"{wg}.fc_bias.weight", f"{kwg}/fc_bias/kernel")
    copy(f"{wg}.fc_bias.bias", f"{kwg}/fc_bias/bias")

    # nn.TransformerEncoder(layer, num_layers=1) -> `encoder.layers.0.*`
    el = f"{wg}.encoder.layers.0"
    kel = f"{kwg}/encoder_layer"
    if f"{el}.self_attn.in_proj_weight" in flat_state_dict:
        out[f"{kel}/in_proj/kernel"] = flat_state_dict[f"{el}.self_attn.in_proj_weight"]
        out[f"{kel}/in_proj/bias"] = flat_state_dict[f"{el}.self_attn.in_proj_bias"]
    copy(f"{el}.self_attn.out_proj.weight", f"{kel}/out_proj/kernel")
    copy(f"{el}.self_attn.out_proj.bias", f"{kel}/out_proj/bias")
    copy(f"{el}.linear1.weight", f"{kel}/linear1/kernel")
    copy(f"{el}.linear1.bias", f"{kel}/linear1/bias")
    copy(f"{el}.linear2.weight", f"{kel}/linear2/kernel")
    copy(f"{el}.linear2.bias", f"{kel}/linear2/bias")
    copy(f"{el}.norm1.weight", f"{kel}/norm1/gamma")
    copy(f"{el}.norm1.bias", f"{kel}/norm1/beta")
    copy(f"{el}.norm2.weight", f"{kel}/norm2/gamma")
    copy(f"{el}.norm2.bias", f"{kel}/norm2/beta")

    # Main lucidrains-style Transformer backbone
    for i in range(depth):
        tp, kp = f"transformer.layers.{i}", f"clay_encoder/block{i}"
        copy(f"{tp}.0.norm.weight", f"{kp}/attn_norm/gamma")
        copy(f"{tp}.0.norm.bias", f"{kp}/attn_norm/beta")
        copy(f"{tp}.0.to_qkv.weight", f"{kp}/to_qkv/kernel")
        copy(f"{tp}.0.to_out.weight", f"{kp}/to_out/kernel")
        copy(f"{tp}.1.net.0.weight", f"{kp}/ff_norm/gamma")
        copy(f"{tp}.1.net.0.bias", f"{kp}/ff_norm/beta")
        copy(f"{tp}.1.net.1.weight", f"{kp}/ff1/kernel")
        copy(f"{tp}.1.net.1.bias", f"{kp}/ff1/bias")
        copy(f"{tp}.1.net.3.weight", f"{kp}/ff2/kernel")
        copy(f"{tp}.1.net.3.bias", f"{kp}/ff2/bias")
    copy("transformer.norm.weight", "clay_encoder/norm/gamma")
    copy("transformer.norm.bias", "clay_encoder/norm/beta")

    return out


def build_clay_identity_mapper():
    """`convert_clay_encoder_state_dict` already produces keys spelled
    exactly as the target Keras weight paths."""
    return lambda k: k
