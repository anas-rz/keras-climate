"""
Mapping for `keras_climate.weather.climax.ClimaX`, targeting Microsoft's
official `1.40625deg.ckpt` checkpoint (see `weights/pretrained.py`'s
`climax_1_40625deg` loader).

Two checkpoint-specific naming quirks, confirmed by directly inspecting
the real checkpoint's `state_dict` (its per-variable/aggregation naming
differs from the current `microsoft/ClimaX` GitHub source, which uses
`var_embed`/`var_query`/`var_agg` - this specific released checkpoint
uses `channel_embed`/`channel_query`/`channel_agg` instead):
  * every key is nested under a `net.` prefix (a PyTorch Lightning
    `LightningModule.net` submodule).
  * the per-variable-aggregation cross-attention's parameter names are
    `channel_embed`/`channel_query`/`channel_agg.*`, not `var_*`.
Everything else (`token_embeds.{v}.proj`, `pos_embed`, `lead_time_embed`,
`blocks.{i}.*`, `norm`, `head.{0,2,4}`) matches this repo's own naming
one-to-one.
"""

import re


def _block_rules(i):
    prefix = f"blocks.{i}"
    kp = f"blocks{i}"
    return [
        (rf"^{prefix}\.norm1\.weight$", f"{kp}/norm1/gamma"),
        (rf"^{prefix}\.norm1\.bias$", f"{kp}/norm1/beta"),
        (rf"^{prefix}\.attn\.qkv\.weight$", f"{kp}/attn/qkv/kernel"),
        (rf"^{prefix}\.attn\.qkv\.bias$", f"{kp}/attn/qkv/bias"),
        (rf"^{prefix}\.attn\.proj\.weight$", f"{kp}/attn/proj/kernel"),
        (rf"^{prefix}\.attn\.proj\.bias$", f"{kp}/attn/proj/bias"),
        (rf"^{prefix}\.norm2\.weight$", f"{kp}/norm2/gamma"),
        (rf"^{prefix}\.norm2\.bias$", f"{kp}/norm2/beta"),
        (rf"^{prefix}\.mlp\.fc1\.weight$", f"{kp}/mlp/fc1/kernel"),
        (rf"^{prefix}\.mlp\.fc1\.bias$", f"{kp}/mlp/fc1/bias"),
        (rf"^{prefix}\.mlp\.fc2\.weight$", f"{kp}/mlp/fc2/kernel"),
        (rf"^{prefix}\.mlp\.fc2\.bias$", f"{kp}/mlp/fc2/bias"),
    ]


def build_climax_mapper(num_vars, depth, decoder_depth):
    rules = [
        (r"^pos_embed$", "pos_embed_layer/pos_embed"),
        (r"^channel_embed$", "var_embed_layer/var_embed"),
        (r"^channel_query$", "var_agg/var_query"),
        (r"^channel_agg\.in_proj_weight$", "var_agg/in_proj_weight"),
        (r"^channel_agg\.in_proj_bias$", "var_agg/in_proj_bias"),
        (r"^channel_agg\.out_proj\.weight$", "var_agg/out_proj/kernel"),
        (r"^channel_agg\.out_proj\.bias$", "var_agg/out_proj/bias"),
        (r"^lead_time_embed\.weight$", "lead_time_embed/kernel"),
        (r"^lead_time_embed\.bias$", "lead_time_embed/bias"),
        (r"^norm\.weight$", "norm/gamma"),
        (r"^norm\.bias$", "norm/beta"),
    ]
    for v in range(num_vars):
        rules += [
            (rf"^token_embeds\.{v}\.proj\.weight$", f"token_embeds/token_embeds{v}/kernel"),
            (rf"^token_embeds\.{v}\.proj\.bias$", f"token_embeds/token_embeds{v}/bias"),
        ]
    for i in range(depth):
        rules += _block_rules(i)
    for i in range(decoder_depth + 1):
        idx = 2 * i
        rules += [
            (rf"^head\.{idx}\.weight$", f"head{idx}/kernel"),
            (rf"^head\.{idx}\.bias$", f"head{idx}/bias"),
        ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
