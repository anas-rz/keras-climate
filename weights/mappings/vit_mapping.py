"""
Generic ViT-style mapping builder. Most of this repo's transformer-based
models (SatMAE, Prithvi, CROMA's per-modality encoders, AnySat) use the
same `TransformerEncoderBlock` from `keras_climate.utils.layers`, which in
turn matches the near-universal HuggingFace/timm ViT naming convention on
the PyTorch side:

    blocks.{i}.norm1.{weight,bias}
    blocks.{i}.attn.qkv.{weight,bias}
    blocks.{i}.attn.proj.{weight,bias}
    blocks.{i}.norm2.{weight,bias}
    blocks.{i}.mlp.fc1.{weight,bias}
    blocks.{i}.mlp.fc2.{weight,bias}
    cls_token / pos_embed / patch_embed.proj.{weight,bias} / norm.{weight,bias}

This module builds the corresponding regex rules for a given Keras model
name prefix + block-name pattern, so you don't have to hand-write 12-24
near-identical block mappings per model.
"""

import re


def build_vit_mapper(keras_prefix, block_name_fn=None, extra_rules=None):
    """
    Args:
        keras_prefix: the Keras model's top-level scope, e.g.
            "satmae_encoder" or "prithvi_classifier/encoder". Should match
            the start of `weight.path` for this model's weights.
        block_name_fn: optional `i -> str` giving the Keras block layer name
            if it differs from the default f"block{i}" used by
            `TransformerEncoderBlock` instances in this repo.
        extra_rules: additional (regex, replacement) rules checked first,
            for model-specific tokens (e.g. SegFormer's Mix-FFN naming, or
            Clay's band/geo embeddings) that don't fit the generic ViT
            pattern below.

    Returns a `torch_key -> keras_key | None` callable.
    """
    block_name_fn = block_name_fn or (lambda i: f"block{i}")

    rules = list(extra_rules or [])
    rules += [
        (r"^cls_token$", f"{keras_prefix}/cls_token"),
        (r"^pos_embed$", f"{keras_prefix}/pos_embed"),
        (r"^patch_embed\.proj\.weight$", f"{keras_prefix}/patch_embed/proj/kernel"),
        (r"^patch_embed\.proj\.bias$", f"{keras_prefix}/patch_embed/proj/bias"),
        (r"^norm\.weight$", f"{keras_prefix}/norm/gamma"),
        (r"^norm\.bias$", f"{keras_prefix}/norm/beta"),
    ]

    compiled_static = [(re.compile(p), r) for p, r in rules]

    block_pattern = re.compile(
        r"^blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)$|"
        r"^blocks\.(\d+)\.attn\.(qkv|proj)\.(weight|bias)$|"
        r"^blocks\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)$"
    )

    def mapper(torch_key):
        for pattern, repl in compiled_static:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)

        m = block_pattern.match(torch_key)
        if not m:
            return None

        if m.group(1) is not None:
            idx, sub, param = m.group(1), m.group(2), m.group(3)
            keras_sub = {"norm1": "norm1", "norm2": "norm2"}[sub]
            param_name = "gamma" if param == "weight" else "beta"
            return f"{keras_prefix}/{block_name_fn(int(idx))}/{keras_sub}/{param_name}"

        if m.group(4) is not None:
            idx, sub, param = m.group(4), m.group(5), m.group(6)
            keras_param = "kernel" if param == "weight" else "bias"
            return f"{keras_prefix}/{block_name_fn(int(idx))}/attn/{sub}/{keras_param}"

        idx, sub, param = m.group(7), m.group(8), m.group(9)
        keras_param = "kernel" if param == "weight" else "bias"
        return f"{keras_prefix}/{block_name_fn(int(idx))}/mlp/{sub}/{keras_param}"

    return mapper
