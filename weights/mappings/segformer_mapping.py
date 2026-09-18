"""
Mapping for `keras_climate.remote_sensing.segformer.SegFormer`, assuming a
source checkpoint using the official NVlabs/SegFormer (`mmseg`-style)
naming convention:

    patch_embed{s}.proj.{weight,bias}          s in 1..4
    patch_embed{s}.norm.{weight,bias}
    block{s}.{i}.norm1.{weight,bias}
    block{s}.{i}.attn.q.{weight,bias}
    block{s}.{i}.attn.kv.{weight,bias}
    block{s}.{i}.attn.sr.{weight,bias}         only when sr_ratio > 1
    block{s}.{i}.attn.norm.{weight,bias}       only when sr_ratio > 1
    block{s}.{i}.attn.proj.{weight,bias}
    block{s}.{i}.norm2.{weight,bias}
    block{s}.{i}.mlp.fc1.{weight,bias}
    block{s}.{i}.mlp.dwconv.dwconv.{weight,bias}
    block{s}.{i}.mlp.fc2.{weight,bias}
    norm{s}.{weight,bias}
    decode_head.linear_c{4,3,2,1}.proj.{weight,bias}
    decode_head.linear_fuse.conv.weight
    decode_head.linear_fuse.bn.{weight,bias,running_mean,running_var}
    decode_head.linear_pred.{weight,bias}

onto `SegFormer(variant=..., name="segformer")`'s weight paths
(`segformer_backbone_stage{s}/...`, `segformer_decode_head_.../...`).

The Mix-FFN's depthwise conv needs an explicit `param_kind="depthwise_kernel"`
override (`build_segformer_param_kind_map`) since its PyTorch shape
`(dim, 1, 3, 3)` is not enough on its own to distinguish it from a regular
Conv2D kernel by shape alone.
"""

import re

import numpy as np


def build_segformer_mapper(cfg, keras_name="segformer", decode_head_prefix="decode_head"):
    """torch_key -> keras_key mapper for one `MIT_CONFIGS[...]` variant."""
    rules = []

    for s in range(1, 5):
        stage_kp = f"{keras_name}_backbone_stage{s}"
        rules += [
            (rf"^patch_embed{s}\.proj\.weight$", f"{stage_kp}/patch_embed/proj/kernel"),
            (rf"^patch_embed{s}\.proj\.bias$", f"{stage_kp}/patch_embed/proj/bias"),
            (rf"^patch_embed{s}\.norm\.weight$", f"{stage_kp}/patch_embed/norm/gamma"),
            (rf"^patch_embed{s}\.norm\.bias$", f"{stage_kp}/patch_embed/norm/beta"),
            (rf"^norm{s}\.weight$", f"{stage_kp}/norm/gamma"),
            (rf"^norm{s}\.bias$", f"{stage_kp}/norm/beta"),
        ]

        depth = cfg["depths"][s - 1]
        for i in range(depth):
            tp = f"block{s}.{i}"
            kp = f"{stage_kp}/block{i}"
            rules += [
                (rf"^{re.escape(tp)}\.norm1\.weight$", f"{kp}/norm1/gamma"),
                (rf"^{re.escape(tp)}\.norm1\.bias$", f"{kp}/norm1/beta"),
                (rf"^{re.escape(tp)}\.attn\.q\.weight$", f"{kp}/attn/q/kernel"),
                (rf"^{re.escape(tp)}\.attn\.q\.bias$", f"{kp}/attn/q/bias"),
                (rf"^{re.escape(tp)}\.attn\.kv\.weight$", f"{kp}/attn/kv/kernel"),
                (rf"^{re.escape(tp)}\.attn\.kv\.bias$", f"{kp}/attn/kv/bias"),
                (rf"^{re.escape(tp)}\.attn\.sr\.weight$", f"{kp}/attn/sr/kernel"),
                (rf"^{re.escape(tp)}\.attn\.sr\.bias$", f"{kp}/attn/sr/bias"),
                (rf"^{re.escape(tp)}\.attn\.norm\.weight$", f"{kp}/attn/sr_norm/gamma"),
                (rf"^{re.escape(tp)}\.attn\.norm\.bias$", f"{kp}/attn/sr_norm/beta"),
                (rf"^{re.escape(tp)}\.attn\.proj\.weight$", f"{kp}/attn/proj/kernel"),
                (rf"^{re.escape(tp)}\.attn\.proj\.bias$", f"{kp}/attn/proj/bias"),
                (rf"^{re.escape(tp)}\.norm2\.weight$", f"{kp}/norm2/gamma"),
                (rf"^{re.escape(tp)}\.norm2\.bias$", f"{kp}/norm2/beta"),
                (rf"^{re.escape(tp)}\.mlp\.fc1\.weight$", f"{kp}/mixffn_fc1/kernel"),
                (rf"^{re.escape(tp)}\.mlp\.fc1\.bias$", f"{kp}/mixffn_fc1/bias"),
                (rf"^{re.escape(tp)}\.mlp\.dwconv\.dwconv\.weight$", f"{kp}/mixffn_dwconv/kernel"),
                (rf"^{re.escape(tp)}\.mlp\.dwconv\.dwconv\.bias$", f"{kp}/mixffn_dwconv/bias"),
                (rf"^{re.escape(tp)}\.mlp\.fc2\.weight$", f"{kp}/mixffn_fc2/kernel"),
                (rf"^{re.escape(tp)}\.mlp\.fc2\.bias$", f"{kp}/mixffn_fc2/bias"),
            ]

    dh = decode_head_prefix
    dh_kp = f"{keras_name}_decode_head"
    for i in (4, 3, 2, 1):
        rules += [
            (rf"^{re.escape(dh)}\.linear_c{i}\.proj\.weight$", f"{dh_kp}_linear_c{i}/kernel"),
            (rf"^{re.escape(dh)}\.linear_c{i}\.proj\.bias$", f"{dh_kp}_linear_c{i}/bias"),
        ]
    rules += [
        (rf"^{re.escape(dh)}\.linear_fuse\.conv\.weight$", f"{dh_kp}_fuse/conv/kernel"),
        (rf"^{re.escape(dh)}\.linear_fuse\.bn\.weight$", f"{dh_kp}_fuse/bn/gamma"),
        (rf"^{re.escape(dh)}\.linear_fuse\.bn\.bias$", f"{dh_kp}_fuse/bn/beta"),
        (rf"^{re.escape(dh)}\.linear_fuse\.bn\.running_mean$", f"{dh_kp}_fuse/bn/moving_mean"),
        (rf"^{re.escape(dh)}\.linear_fuse\.bn\.running_var$", f"{dh_kp}_fuse/bn/moving_variance"),
        (rf"^{re.escape(dh)}\.linear_pred\.weight$", f"{dh_kp}_classifier/kernel"),
        (rf"^{re.escape(dh)}\.linear_pred\.bias$", f"{dh_kp}_classifier/bias"),
    ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper


def convert_hf_segformer_state_dict(hf_state_dict, cfg):
    """Translates a HuggingFace `transformers.SegformerForSemanticSegmentation`
    state_dict (e.g. `nvidia/segformer-b0-finetuned-ade-512-512`) into the
    official NVlabs naming that `build_segformer_mapper` expects, so the
    same mapper/converter works for both source formats.

    HF's naming differs from the official repo in two ways that matter
    here: (1) every submodule is renamed (`segformer.stages.{s}.
    patch_embeddings...` / `attention.q_proj/k_proj/v_proj/o_proj...` /
    `decode_head.linear_projections.{i}...` / `decode_head.batch_norm...`
    / `decode_head.classifier...`), and (2) HF splits the fused `kv`
    projection into two separate `k_proj`/`v_proj` Linears - those have to
    be concatenated back into one `(2*dim, dim)` tensor since this repo's
    `SpatialReductionAttention` (matching the official architecture) uses
    a single fused `kv` Dense.
    """
    out = {}
    depths = cfg["depths"]

    for s in range(4):
        prefix = f"segformer.stages.{s}"
        out[f"patch_embed{s+1}.proj.weight"] = hf_state_dict[f"{prefix}.patch_embeddings.proj.weight"]
        out[f"patch_embed{s+1}.proj.bias"] = hf_state_dict[f"{prefix}.patch_embeddings.proj.bias"]
        out[f"patch_embed{s+1}.norm.weight"] = hf_state_dict[f"{prefix}.patch_embeddings.layer_norm.weight"]
        out[f"patch_embed{s+1}.norm.bias"] = hf_state_dict[f"{prefix}.patch_embeddings.layer_norm.bias"]
        out[f"norm{s+1}.weight"] = hf_state_dict[f"{prefix}.layer_norm.weight"]
        out[f"norm{s+1}.bias"] = hf_state_dict[f"{prefix}.layer_norm.bias"]

        for i in range(depths[s]):
            bp = f"{prefix}.blocks.{i}"
            tp = f"block{s+1}.{i}"
            out[f"{tp}.norm1.weight"] = hf_state_dict[f"{bp}.layernorm_before.weight"]
            out[f"{tp}.norm1.bias"] = hf_state_dict[f"{bp}.layernorm_before.bias"]
            out[f"{tp}.attn.q.weight"] = hf_state_dict[f"{bp}.attention.q_proj.weight"]
            out[f"{tp}.attn.q.bias"] = hf_state_dict[f"{bp}.attention.q_proj.bias"]
            out[f"{tp}.attn.kv.weight"] = np.concatenate(
                [hf_state_dict[f"{bp}.attention.k_proj.weight"],
                 hf_state_dict[f"{bp}.attention.v_proj.weight"]], axis=0)
            out[f"{tp}.attn.kv.bias"] = np.concatenate(
                [hf_state_dict[f"{bp}.attention.k_proj.bias"],
                 hf_state_dict[f"{bp}.attention.v_proj.bias"]], axis=0)
            out[f"{tp}.attn.proj.weight"] = hf_state_dict[f"{bp}.attention.o_proj.weight"]
            out[f"{tp}.attn.proj.bias"] = hf_state_dict[f"{bp}.attention.o_proj.bias"]
            sr_key = f"{bp}.attention.sequence_reduction.sequence_reduction.weight"
            if sr_key in hf_state_dict:
                out[f"{tp}.attn.sr.weight"] = hf_state_dict[sr_key]
                out[f"{tp}.attn.sr.bias"] = hf_state_dict[
                    f"{bp}.attention.sequence_reduction.sequence_reduction.bias"]
                out[f"{tp}.attn.norm.weight"] = hf_state_dict[
                    f"{bp}.attention.sequence_reduction.layer_norm.weight"]
                out[f"{tp}.attn.norm.bias"] = hf_state_dict[
                    f"{bp}.attention.sequence_reduction.layer_norm.bias"]
            out[f"{tp}.norm2.weight"] = hf_state_dict[f"{bp}.layernorm_after.weight"]
            out[f"{tp}.norm2.bias"] = hf_state_dict[f"{bp}.layernorm_after.bias"]
            out[f"{tp}.mlp.fc1.weight"] = hf_state_dict[f"{bp}.mlp.fc1.weight"]
            out[f"{tp}.mlp.fc1.bias"] = hf_state_dict[f"{bp}.mlp.fc1.bias"]
            out[f"{tp}.mlp.dwconv.dwconv.weight"] = hf_state_dict[f"{bp}.mlp.dwconv.dwconv.weight"]
            out[f"{tp}.mlp.dwconv.dwconv.bias"] = hf_state_dict[f"{bp}.mlp.dwconv.dwconv.bias"]
            out[f"{tp}.mlp.fc2.weight"] = hf_state_dict[f"{bp}.mlp.fc2.weight"]
            out[f"{tp}.mlp.fc2.bias"] = hf_state_dict[f"{bp}.mlp.fc2.bias"]

    for i in (1, 2, 3, 4):
        out[f"decode_head.linear_c{i}.proj.weight"] = hf_state_dict[f"decode_head.linear_projections.{i-1}.proj.weight"]
        out[f"decode_head.linear_c{i}.proj.bias"] = hf_state_dict[f"decode_head.linear_projections.{i-1}.proj.bias"]

    out["decode_head.linear_fuse.conv.weight"] = hf_state_dict["decode_head.linear_fuse.weight"]
    out["decode_head.linear_fuse.bn.weight"] = hf_state_dict["decode_head.batch_norm.weight"]
    out["decode_head.linear_fuse.bn.bias"] = hf_state_dict["decode_head.batch_norm.bias"]
    out["decode_head.linear_fuse.bn.running_mean"] = hf_state_dict["decode_head.batch_norm.running_mean"]
    out["decode_head.linear_fuse.bn.running_var"] = hf_state_dict["decode_head.batch_norm.running_var"]
    out["decode_head.linear_pred.weight"] = hf_state_dict["decode_head.classifier.weight"]
    out["decode_head.linear_pred.bias"] = hf_state_dict["decode_head.classifier.bias"]

    return out


def build_segformer_param_kind_map(cfg, keras_name="segformer"):
    """`keras_weight_name -> "depthwise_kernel"` overrides for every Mix-FFN
    depthwise conv kernel in the given `MIT_CONFIGS[...]` variant - required
    because a PyTorch depthwise kernel's shape `(dim, 1, 3, 3)` alone isn't
    enough for `infer_transpose` to distinguish it from a regular conv."""
    out = {}
    for s in range(1, 5):
        depth = cfg["depths"][s - 1]
        for i in range(depth):
            out[f"{keras_name}_backbone_stage{s}/block{i}/mixffn_dwconv/kernel"] = "depthwise_kernel"
    return out
