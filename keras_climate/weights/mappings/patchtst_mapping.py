import re


def build_patchtst_mapper(depth, use_revin=True):
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
