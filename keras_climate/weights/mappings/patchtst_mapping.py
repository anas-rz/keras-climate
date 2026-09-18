"""
Mapping for `keras_climate.forecasting.patchtst.PatchTST`.

Important caveat: the HuggingFace/IBM-Granite PatchTST checkpoints (the
only genuinely downloadable public PatchTST weights found) use a
substantially different architecture from both this repo and the original
research implementation - a CLS token, sincos (not learned) positional
encoding, and BatchNorm (not LayerNorm) inside the transformer encoder -
so it isn't targeted here. This mapper instead targets this repo's own
architecture (closely following the *original* yuqinie98/PatchTST
research repo's patching/RevIN behavior, including its default
`padding_patch="end"` replication-padding and its non-obvious
`gamma + eps**2` RevIN denormalize formula - see patchtst.py), for
porting weights from a from-scratch model trained with this exact
architecture.

Each transformer block follows the same timm-style naming (`norm1`,
`attn.qkv`, `attn.proj`, `norm2`, `mlp.fc1`, `mlp.fc2`) as most other
blocks in this repo, but - like Earthformer's blocks - the Keras layer
names are flat (`encoder_block{i}`, not a nested `encoder/blocks.{i}`
scope), so the block rules are written directly rather than through
`build_vit_mapper`.
"""

import re


def build_patchtst_mapper(depth, use_revin=True):
    """torch_key -> keras_key mapper, assuming a source checkpoint using
    this repo's own naming: `revin.affine_weight/affine_bias`,
    `patch_proj.{weight,bias}`, `pos_embed` (a raw `(num_patches, d_model)`
    parameter, not an `nn.Embedding`), `blocks.{i}...` (standard
    ViT block), `encoder_norm.{weight,bias}`, `forecast_head.{weight,bias}`."""
    rules = [
        (r"^patch_proj\.weight$", "patch_proj/kernel"),
        (r"^patch_proj\.bias$", "patch_proj/bias"),
        (r"^pos_embed$", "pos_embed/embeddings"),
        (r"^encoder_norm\.weight$", "encoder_norm/gamma"),
        (r"^encoder_norm\.bias$", "encoder_norm/beta"),
        (r"^forecast_head\.weight$", "forecast_head/kernel"),
        (r"^forecast_head\.bias$", "forecast_head/bias"),
    ]
    if use_revin:
        rules += [
            (r"^revin\.affine_weight$", "revin/gamma"),
            (r"^revin\.affine_bias$", "revin/beta"),
        ]

    for i in range(depth):
        tp, kp = f"blocks.{i}", f"encoder_block{i}"
        rules += [
            (rf"^{tp}\.norm1\.weight$", f"{kp}/norm1/gamma"),
            (rf"^{tp}\.norm1\.bias$", f"{kp}/norm1/beta"),
            (rf"^{tp}\.attn\.qkv\.weight$", f"{kp}/attn/qkv/kernel"),
            (rf"^{tp}\.attn\.qkv\.bias$", f"{kp}/attn/qkv/bias"),
            (rf"^{tp}\.attn\.proj\.weight$", f"{kp}/attn/proj/kernel"),
            (rf"^{tp}\.attn\.proj\.bias$", f"{kp}/attn/proj/bias"),
            (rf"^{tp}\.norm2\.weight$", f"{kp}/norm2/gamma"),
            (rf"^{tp}\.norm2\.bias$", f"{kp}/norm2/beta"),
            (rf"^{tp}\.mlp\.fc1\.weight$", f"{kp}/mlp/fc1/kernel"),
            (rf"^{tp}\.mlp\.fc1\.bias$", f"{kp}/mlp/fc1/bias"),
            (rf"^{tp}\.mlp\.fc2\.weight$", f"{kp}/mlp/fc2/kernel"),
            (rf"^{tp}\.mlp\.fc2\.bias$", f"{kp}/mlp/fc2/bias"),
        ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
