"""
Mapping for `keras_climate.foundation.anysat_release.AnySatRelease` - the
faithful port of the *officially released* AnySat architecture (as opposed
to `anysat_mapping.py`, which targets this repo's own simpler design; see
`foundation/anysat_release.py`'s module docstring).

Targets the real `g-astruc/AnySat` checkpoint's own key naming one-to-one
(`cls_token`, `projector_{modality}...`, `spatial_encoder...`,
`blocks.{i}...`), not a from-scratch reference.
"""

import re


def build_anysat_release_mapper(modalities, depth, keras_prefix="anysat_release"):
    """torch_key -> keras_key for `AnySatRelease`, given the exact
    `modalities` list and `depth` (number of plain global blocks; the
    checkpoint's `blocks` list has `depth + 1` entries, the last being the
    CrossBlockMulti pooling block) the Keras model was built with."""

    def mapper(torch_key):
        if torch_key == "cls_token":
            return f"{keras_prefix}/cls_token"

        m = re.match(r"^projector_(.+?)\.(.*)$", torch_key)
        if m:
            modality, rest = m.group(1), m.group(2)
            if modality not in modalities:
                return None
            mapped = _map_projector(modality, rest)
            if mapped is None:
                return None
            return f"{keras_prefix}/projector_{modality}/{mapped}"

        m = re.match(r"^spatial_encoder\.(.*)$", torch_key)
        if m:
            mapped = _map_local_encoder(m.group(1))
            if mapped is None:
                return None
            return f"{keras_prefix}/spatial_encoder/{mapped}"

        m = re.match(r"^blocks\.(\d+)\.(.*)$", torch_key)
        if m:
            idx, rest = int(m.group(1)), m.group(2)
            if idx == depth:
                mapped = _map_cross_block(rest)
                if mapped is None:
                    return None
                return f"{keras_prefix}/block_cross/{mapped}"
            mapped = _map_plain_block(rest)
            if mapped is None:
                return None
            return f"{keras_prefix}/block{idx}/{mapped}"

        return None

    return mapper


def _norm_pair(prefix, rest, out_name):
    if rest == f"{prefix}.weight":
        return f"{out_name}/gamma"
    if rest == f"{prefix}.bias":
        return f"{out_name}/beta"
    return None


def _lin_pair(prefix, rest, out_name):
    if rest == f"{prefix}.weight":
        return f"{out_name}/kernel"
    if rest == f"{prefix}.bias":
        return f"{out_name}/bias"
    return None


def _map_projector(modality, rest):
    if rest == "pad_parameter":
        return None  # training-time masking helper, no Keras counterpart

    # image projector: patch_embed.{weight}, mlp.{0,1,3,4}.{weight,bias}
    if rest == "patch_embed.weight":
        return "patch_embed/kernel"
    m = re.match(r"^mlp\.(\d+)\.(weight|bias)$", rest)
    if m:
        idx = int(m.group(1))
        # Sequential [Linear, LayerNorm, ReLU] x2 -> Linear @ {0,3}, LN @ {1,4}
        if idx == 0:
            return _lin_pair("mlp.0", rest, "mlp_lin0")
        if idx == 1:
            return _norm_pair("mlp.1", rest, "mlp_ln0")
        if idx == 3:
            return _lin_pair("mlp.3", rest, "mlp_lin1")
        if idx == 4:
            return _norm_pair("mlp.4", rest, "mlp_ln1")
        return None

    # time-series (LTAE) projector, nested under `patch_embed.`
    pe = "patch_embed."
    if not rest.startswith(pe):
        return None
    r = rest[len(pe):]

    if r == "attention_heads.Q":
        return "Q"
    out = _lin_pair("attention_heads.fc1_k", r, "fc1_k")
    if out:
        return out

    m = re.match(r"^inconv\.(\d+)\.(weight|bias)$", r)
    if m:
        idx = int(m.group(1))
        # Sequential [Linear, GroupNorm, ReLU, Dropout] x N -> Linear @ 4k, GN @ 4k+1
        if idx % 4 == 0:
            return _lin_pair(f"inconv.{idx}", r, f"inconv_lin{idx // 4}")
        if idx % 4 == 1:
            return _norm_pair(f"inconv.{idx}", r, f"inconv_gn{idx // 4}")
        return None

    out = _norm_pair("in_norm", r, "in_norm")
    if out:
        return out

    m = re.match(r"^mlp\.(\d+)\.(weight|bias)$", r)
    if m:
        idx = int(m.group(1))
        if idx == 0:
            return _lin_pair("mlp.0", r, "mlp_lin0")
        if idx == 1:
            return _norm_pair("mlp.1", r, "mlp_gn0")
        return None

    out = _norm_pair("out_norm", r, "out_norm")
    if out:
        return out

    return None


def _map_local_encoder(rest):
    if rest == "cls_token":
        return "cls_token"
    if rest in ("predictor_norm.weight", "predictor_norm.bias"):
        return _norm_pair("predictor_norm", rest, "norm")

    m = re.match(r"^predictor_blocks\.(\d+)\.(.*)$", rest)
    if not m:
        return None
    idx, r = int(m.group(1)), m.group(2)
    prefix = f"block{idx}"

    out = _norm_pair("norm1", r, f"{prefix}_norm1")
    if out:
        return out
    out = _norm_pair("norm2", r, f"{prefix}_norm2")
    if out:
        return out
    out = _lin_pair("attn.qkv", r, f"{prefix}_attn_qkv")
    if out:
        return out
    out = _lin_pair("attn.proj", r, f"{prefix}_attn_proj")
    if out:
        return out
    out = _lin_pair("mlp.fc1", r, f"{prefix}_mlp_fc1")
    if out:
        return out
    out = _lin_pair("mlp.fc2", r, f"{prefix}_mlp_fc2")
    if out:
        return out
    if r == "attn.rpe_k.lookup_table_weight":
        return f"{prefix}_rpe_k_weight"
    return None


def _map_plain_block(rest):
    out = _norm_pair("norm1", rest, "norm1")
    if out:
        return out
    out = _norm_pair("norm2", rest, "norm2")
    if out:
        return out
    out = _lin_pair("attn.qkv", rest, "attn_qkv")
    if out:
        return out
    out = _lin_pair("attn.proj", rest, "attn_proj")
    if out:
        return out
    out = _lin_pair("mlp.fc1", rest, "mlp_fc1")
    if out:
        return out
    out = _lin_pair("mlp.fc2", rest, "mlp_fc2")
    if out:
        return out
    return None


def _map_cross_block(rest):
    out = _norm_pair("norm1", rest, "norm1")
    if out:
        return out
    out = _norm_pair("norm2", rest, "norm2")
    if out:
        return out
    out = _lin_pair("attn.wk", rest, "attn_wk")
    if out:
        return out
    out = _lin_pair("attn.wv", rest, "attn_wv")
    if out:
        return out
    out = _lin_pair("attn.proj", rest, "attn_proj")
    if out:
        return out
    out = _lin_pair("mlp.fc1", rest, "mlp_fc1")
    if out:
        return out
    out = _lin_pair("mlp.fc2", rest, "mlp_fc2")
    if out:
        return out
    if rest == "attn.q_learned":
        return "attn_q_learned"
    if rest == "attn.rpe_k.lookup_table_weight":
        return "attn_rpe_k_weight"
    return None
