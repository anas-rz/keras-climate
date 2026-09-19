import re

import numpy as np


def load_croma_checkpoint(path):
    import torch

    raw = torch.load(path, map_location="cpu")
    flat = {}
    for submodule, sub_sd in raw.items():
        for k, v in sub_sd.items():
            if k.endswith("num_batches_tracked"):
                continue
            flat[f"{submodule}.{k}"] = (
                v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            )
    return flat


def _self_block_rules(torch_prefix, keras_prefix, depth):
    out = {}
    for i in range(depth):
        tp, kp = f"{torch_prefix}.transformer.layers.{i}", f"{keras_prefix}/block{i}"
        out[f"{tp}.0.input_norm.weight"] = f"{kp}/attn/input_norm/gamma"
        out[f"{tp}.0.input_norm.bias"] = f"{kp}/attn/input_norm/beta"
        out[f"{tp}.0.to_qkv.weight"] = f"{kp}/attn/to_qkv/kernel"
        out[f"{tp}.0.to_out.weight"] = f"{kp}/attn/to_out/kernel"
        out[f"{tp}.0.to_out.bias"] = f"{kp}/attn/to_out/bias"
        out[f"{tp}.1.input_norm.weight"] = f"{kp}/ffn/input_norm/gamma"
        out[f"{tp}.1.input_norm.bias"] = f"{kp}/ffn/input_norm/beta"
        out[f"{tp}.1.net.0.weight"] = f"{kp}/ffn/fc1/kernel"
        out[f"{tp}.1.net.0.bias"] = f"{kp}/ffn/fc1/bias"
        out[f"{tp}.1.net.3.weight"] = f"{kp}/ffn/fc2/kernel"
        out[f"{tp}.1.net.3.bias"] = f"{kp}/ffn/fc2/bias"
    out[f"{torch_prefix}.transformer.norm_out.weight"] = (
        f"{keras_prefix}/norm_out/gamma"
    )
    out[f"{torch_prefix}.transformer.norm_out.bias"] = f"{keras_prefix}/norm_out/beta"
    return out


def _gap_ffn_rules(torch_prefix, keras_prefix):
    return {
        f"{torch_prefix}.0.weight": f"{keras_prefix}/norm/gamma",
        f"{torch_prefix}.0.bias": f"{keras_prefix}/norm/beta",
        f"{torch_prefix}.1.weight": f"{keras_prefix}/fc1/kernel",
        f"{torch_prefix}.1.bias": f"{keras_prefix}/fc1/bias",
        f"{torch_prefix}.3.weight": f"{keras_prefix}/fc2/kernel",
        f"{torch_prefix}.3.bias": f"{keras_prefix}/fc2/bias",
    }


def _cross_block_rules(torch_prefix, keras_prefix, depth):
    out = {}
    for i in range(depth):
        tp, kp = f"{torch_prefix}.layers.{i}", f"{keras_prefix}/block{i}"
        out[f"{tp}.0.input_norm.weight"] = f"{kp}/self_attn/input_norm/gamma"
        out[f"{tp}.0.input_norm.bias"] = f"{kp}/self_attn/input_norm/beta"
        out[f"{tp}.0.to_qkv.weight"] = f"{kp}/self_attn/to_qkv/kernel"
        out[f"{tp}.0.to_out.weight"] = f"{kp}/self_attn/to_out/kernel"
        out[f"{tp}.0.to_out.bias"] = f"{kp}/self_attn/to_out/bias"

        out[f"{tp}.1.input_norm.weight"] = f"{kp}/cross_attn/input_norm/gamma"
        out[f"{tp}.1.input_norm.bias"] = f"{kp}/cross_attn/input_norm/beta"
        out[f"{tp}.1.to_q.weight"] = f"{kp}/cross_attn/to_q/kernel"
        out[f"{tp}.1.to_k.weight"] = f"{kp}/cross_attn/to_k/kernel"
        out[f"{tp}.1.to_v.weight"] = f"{kp}/cross_attn/to_v/kernel"
        out[f"{tp}.1.to_out.weight"] = f"{kp}/cross_attn/to_out/kernel"
        out[f"{tp}.1.to_out.bias"] = f"{kp}/cross_attn/to_out/bias"

        out[f"{tp}.2.input_norm.weight"] = f"{kp}/ffn/input_norm/gamma"
        out[f"{tp}.2.input_norm.bias"] = f"{kp}/ffn/input_norm/beta"
        out[f"{tp}.2.net.0.weight"] = f"{kp}/ffn/fc1/kernel"
        out[f"{tp}.2.net.0.bias"] = f"{kp}/ffn/fc1/bias"
        out[f"{tp}.2.net.3.weight"] = f"{kp}/ffn/fc2/kernel"
        out[f"{tp}.2.net.3.bias"] = f"{kp}/ffn/fc2/bias"
    out[f"{torch_prefix}.norm_out.weight"] = f"{keras_prefix}/norm_out/gamma"
    out[f"{torch_prefix}.norm_out.bias"] = f"{keras_prefix}/norm_out/beta"
    return out


def convert_croma_state_dict(flat_state_dict, encoder_depth=12):
    key_map = {}
    key_map["s1_encoder.linear_input.weight"] = "s1_encoder/linear_input/kernel"
    key_map["s1_encoder.linear_input.bias"] = "s1_encoder/linear_input/bias"
    key_map.update(_self_block_rules("s1_encoder", "s1_encoder", encoder_depth // 2))

    key_map["s2_encoder.linear_input.weight"] = "s2_encoder/linear_input/kernel"
    key_map["s2_encoder.linear_input.bias"] = "s2_encoder/linear_input/bias"
    key_map.update(_self_block_rules("s2_encoder", "s2_encoder", encoder_depth))

    key_map.update(_gap_ffn_rules("s1_GAP_FFN", "GAP_FFN_s1"))
    key_map.update(_gap_ffn_rules("s2_GAP_FFN", "GAP_FFN_s2"))

    key_map.update(
        _cross_block_rules("joint_encoder", "cross_encoder", encoder_depth // 2)
    )

    out = {}
    for torch_key, keras_key in key_map.items():
        if torch_key in flat_state_dict:
            out[keras_key] = flat_state_dict[torch_key]
    return out


def build_croma_identity_mapper():
    return lambda k: k
