"""
Mapping for `keras_climate.weather.earthformer.Earthformer`.

Important caveat: the *officially released* Earthformer checkpoints
(amazon-science/earth-forecasting-transformer, hosted at
`earthformer.s3.amazonaws.com/pretrained_checkpoints/...`) are no longer
publicly reachable (the bucket now returns 403 Forbidden for anonymous
requests) - so there is no live official checkpoint to target. This mapper
instead targets this repo's own architecture (a simplified hierarchical
Cuboid-Attention U-Net capturing Earthformer's core idea, not a byte-exact
reimplementation of the paper's full CuboidTransformer), for porting
weights from a from-scratch model trained with this exact architecture.

Each `CuboidTransformerBlock` follows the same timm-style ViT block naming
(`norm1`, `attn.qkv`, `attn.proj`, `norm2`, `mlp.fc1`, `mlp.fc2`) as most
other transformer blocks in this repo, but note the Keras layer names are
*flat*, underscore-joined per stage (`enc_stage{i}_block{j}`,
`dec_stage{i}_block{j}`) rather than nested scopes - so this mapper writes
the block rules directly instead of reusing `build_vit_mapper` (which
always joins its prefix and block name with "/").
"""

import re


def _block_rules(torch_stage_prefix, keras_stage_prefix, depth):
    """`{torch_stage_prefix}.blocks.{j}...` -> `{keras_stage_prefix}_block{j}/...`"""
    rules = []
    for j in range(depth):
        tp, kp = f"{torch_stage_prefix}.blocks.{j}", f"{keras_stage_prefix}_block{j}"
        rules += [
            (rf"^{re.escape(tp)}\.norm1\.weight$", f"{kp}/norm1/gamma"),
            (rf"^{re.escape(tp)}\.norm1\.bias$", f"{kp}/norm1/beta"),
            (rf"^{re.escape(tp)}\.attn\.qkv\.weight$", f"{kp}/attn/qkv/kernel"),
            (rf"^{re.escape(tp)}\.attn\.qkv\.bias$", f"{kp}/attn/qkv/bias"),
            (rf"^{re.escape(tp)}\.attn\.proj\.weight$", f"{kp}/attn/proj/kernel"),
            (rf"^{re.escape(tp)}\.attn\.proj\.bias$", f"{kp}/attn/proj/bias"),
            (rf"^{re.escape(tp)}\.norm2\.weight$", f"{kp}/norm2/gamma"),
            (rf"^{re.escape(tp)}\.norm2\.bias$", f"{kp}/norm2/beta"),
            (rf"^{re.escape(tp)}\.mlp\.fc1\.weight$", f"{kp}/mlp/fc1/kernel"),
            (rf"^{re.escape(tp)}\.mlp\.fc1\.bias$", f"{kp}/mlp/fc1/bias"),
            (rf"^{re.escape(tp)}\.mlp\.fc2\.weight$", f"{kp}/mlp/fc2/kernel"),
            (rf"^{re.escape(tp)}\.mlp\.fc2\.bias$", f"{kp}/mlp/fc2/bias"),
        ]
    return rules


def build_earthformer_mapper(num_enc_stages, num_dec_stages, enc_depths=None, dec_depths=None):
    """torch_key -> keras_key mapper, assuming a source checkpoint using
    this repo's own naming convention: `stem.conv.{weight,bias}`,
    `enc_stage{i}.blocks.{j}...` / `dec_stage{i}.blocks.{j}...` (standard
    ViT block, see module docstring), `downsample{i}.conv.{weight,bias}`,
    `upsample{i}.conv.{weight,bias}`, `skip_proj{i}.conv.{weight,bias}`,
    `time_channel_proj.{weight,bias}`.

    `enc_depths`/`dec_depths` (defaulting to 1 block per stage if omitted)
    must match how many `CuboidTransformerBlock`s each stage of the target
    `Earthformer(...)` model actually has."""
    enc_depths = enc_depths or [1] * num_enc_stages
    dec_depths = dec_depths or [1] * num_dec_stages

    rules = [
        (r"^stem\.conv\.weight$", "stem/conv/kernel"),
        (r"^stem\.conv\.bias$", "stem/conv/bias"),
        (r"^time_channel_proj\.weight$", "time_channel_proj/kernel"),
        (r"^time_channel_proj\.bias$", "time_channel_proj/bias"),
    ]
    for i in range(num_enc_stages - 1):
        rules += [
            (rf"^downsample{i}\.conv\.weight$", f"downsample{i}/conv/kernel"),
            (rf"^downsample{i}\.conv\.bias$", f"downsample{i}/conv/bias"),
        ]
    for i in range(num_dec_stages):
        rules += [
            (rf"^upsample{i}\.conv\.weight$", f"upsample{i}/conv/kernel"),
            (rf"^upsample{i}\.conv\.bias$", f"upsample{i}/conv/bias"),
            (rf"^skip_proj{i}\.conv\.weight$", f"skip_proj{i}/conv/kernel"),
            (rf"^skip_proj{i}\.conv\.bias$", f"skip_proj{i}/conv/bias"),
        ]

    for i in range(num_enc_stages):
        rules += _block_rules(f"enc_stage{i}", f"enc_stage{i}", enc_depths[i])
    for i in range(num_dec_stages):
        rules += _block_rules(f"dec_stage{i}", f"dec_stage{i}", dec_depths[i])

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
