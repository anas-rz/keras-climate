import re

from keras_climate.weights.mappings.vit_mapping import build_vit_mapper


EXTRA_EMBED_KEYS = {
    "multispectral": [(r"^channel_embed$", "{prefix}/group_embed")],
    "temporal": [(r"^temporal_embed$", "{prefix}/temporal_embed")],
}


def build_satmae_encoder_mapper(keras_prefix="satmae_encoder", mode="single"):
    extra_rules = [(p, r.format(prefix=keras_prefix)) for p, r in EXTRA_EMBED_KEYS.get(mode, [])]
    return build_vit_mapper(keras_prefix, extra_rules=extra_rules)


def build_satmae_decoder_mapper(keras_prefix="satmae_decoder", torch_prefix=""):
    tp = re.escape(torch_prefix)
    rules = [
        (rf"^{tp}decoder_embed\.weight$", f"{keras_prefix}/decoder_embed/kernel"),
        (rf"^{tp}decoder_embed\.bias$", f"{keras_prefix}/decoder_embed/bias"),
        (rf"^{tp}mask_token$", f"{keras_prefix}/mask_token"),
        (rf"^{tp}decoder_pos_embed$", f"{keras_prefix}/decoder_pos_embed"),
        (rf"^{tp}decoder_norm\.weight$", f"{keras_prefix}/norm/gamma"),
        (rf"^{tp}decoder_norm\.bias$", f"{keras_prefix}/norm/beta"),
        (rf"^{tp}decoder_pred\.weight$", f"{keras_prefix}/pred/kernel"),
        (rf"^{tp}decoder_pred\.bias$", f"{keras_prefix}/pred/bias"),
    ]

    block_pattern = re.compile(
        rf"^{tp}decoder_blocks\.(\d+)\.(norm1|norm2)\.(weight|bias)$|"
        rf"^{tp}decoder_blocks\.(\d+)\.attn\.(qkv|proj)\.(weight|bias)$|"
        rf"^{tp}decoder_blocks\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)$"
    )
    compiled_static = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled_static:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)

        m = block_pattern.match(torch_key)
        if not m:
            return None

        if m.group(1) is not None:
            idx, sub, param = m.group(1), m.group(2), m.group(3)
            param_name = "gamma" if param == "weight" else "beta"
            return f"{keras_prefix}/block{int(idx)}/{sub}/{param_name}"

        if m.group(4) is not None:
            idx, sub, param = m.group(4), m.group(5), m.group(6)
            keras_param = "kernel" if param == "weight" else "bias"
            return f"{keras_prefix}/block{int(idx)}/attn/{sub}/{keras_param}"

        idx, sub, param = m.group(7), m.group(8), m.group(9)
        keras_param = "kernel" if param == "weight" else "bias"
        return f"{keras_prefix}/block{int(idx)}/mlp/{sub}/{keras_param}"

    return mapper


def build_satmae_mapper(encoder_prefix="satmae_encoder", decoder_prefix="satmae_decoder", mode="single"):
    encoder_mapper = build_satmae_encoder_mapper(encoder_prefix, mode)
    decoder_mapper = build_satmae_decoder_mapper(decoder_prefix)

    def mapper(torch_key):
        result = encoder_mapper(torch_key)
        if result is not None:
            return result
        return decoder_mapper(torch_key)

    return mapper
