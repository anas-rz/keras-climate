import re


def _conv_bn_act_rules(torch_prefix, keras_prefix):
    tp = re.escape(torch_prefix)
    return [
        (rf"^{tp}\.conv\.weight$", f"{keras_prefix}/conv/kernel"),
        (rf"^{tp}\.bn\.weight$", f"{keras_prefix}/bn/gamma"),
        (rf"^{tp}\.bn\.bias$", f"{keras_prefix}/bn/beta"),
        (rf"^{tp}\.bn\.running_mean$", f"{keras_prefix}/bn/moving_mean"),
        (rf"^{tp}\.bn\.running_var$", f"{keras_prefix}/bn/moving_variance"),
    ]


def _swin_block_rules(torch_prefix, keras_prefix):
    tp = re.escape(torch_prefix)
    return [
        (rf"^{tp}\.norm1\.weight$", f"{keras_prefix}/norm1/gamma"),
        (rf"^{tp}\.norm1\.bias$", f"{keras_prefix}/norm1/beta"),
        (rf"^{tp}\.attn\.qkv\.weight$", f"{keras_prefix}/attn/qkv/kernel"),
        (rf"^{tp}\.attn\.qkv\.bias$", f"{keras_prefix}/attn/qkv/bias"),
        (rf"^{tp}\.attn\.proj\.weight$", f"{keras_prefix}/attn/proj/kernel"),
        (rf"^{tp}\.attn\.proj\.bias$", f"{keras_prefix}/attn/proj/bias"),
        (rf"^{tp}\.attn\.relative_position_bias_table$",
         f"{keras_prefix}/attn/relative_position_bias_table"),
        (rf"^{tp}\.norm2\.weight$", f"{keras_prefix}/norm2/gamma"),
        (rf"^{tp}\.norm2\.bias$", f"{keras_prefix}/norm2/beta"),
        (rf"^{tp}\.mlp\.fc1\.weight$", f"{keras_prefix}/mlp/fc1/kernel"),
        (rf"^{tp}\.mlp\.fc1\.bias$", f"{keras_prefix}/mlp/fc1/bias"),
        (rf"^{tp}\.mlp\.fc2\.weight$", f"{keras_prefix}/mlp/fc2/kernel"),
        (rf"^{tp}\.mlp\.fc2\.bias$", f"{keras_prefix}/mlp/fc2/bias"),
    ]


def build_ringmo_encoder_mapper(depths, keras_prefix="ringmo_encoder"):
    rules = []
    rules += _conv_bn_act_rules("patch_embed.conv1", f"{keras_prefix}/patch_embed/conv1")
    rules += _conv_bn_act_rules("patch_embed.conv2", f"{keras_prefix}/patch_embed/conv2")
    rules += [
        (r"^patch_embed\.conv3\.weight$", f"{keras_prefix}/patch_embed/conv3/kernel"),
        (r"^patch_embed\.conv3\.bias$", f"{keras_prefix}/patch_embed/conv3/bias"),
        (r"^patch_embed\.norm\.weight$", f"{keras_prefix}/patch_embed/norm/gamma"),
        (r"^patch_embed\.norm\.bias$", f"{keras_prefix}/patch_embed/norm/beta"),
        (r"^norm\.weight$", f"{keras_prefix}/norm/gamma"),
        (r"^norm\.bias$", f"{keras_prefix}/norm/beta"),
    ]

    for i, depth in enumerate(depths):
        for j in range(depth):
            rules += _swin_block_rules(f"stages.{i}.blocks.{j}", f"{keras_prefix}/stage{i}/block{j}")
        if i < len(depths) - 1:
            rules += [
                (rf"^stages\.{i}\.downsample\.reduction\.weight$",
                 f"{keras_prefix}/stage{i}/downsample/reduction/kernel"),
                (rf"^stages\.{i}\.downsample\.norm\.weight$",
                 f"{keras_prefix}/stage{i}/downsample/norm/gamma"),
                (rf"^stages\.{i}\.downsample\.norm\.bias$",
                 f"{keras_prefix}/stage{i}/downsample/norm/beta"),
            ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper


def build_ringmo_decoder_mapper(keras_prefix="ringmo_decoder"):
    rules = [
        (r"^decoder\.weight$", f"{keras_prefix}/conv/kernel"),
        (r"^decoder\.bias$", f"{keras_prefix}/conv/bias"),
    ]
    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper


def build_ringmo_mapper(depths, encoder_prefix="ringmo_encoder", decoder_prefix="ringmo_decoder"):
    encoder_mapper = build_ringmo_encoder_mapper(depths, encoder_prefix)
    decoder_mapper = build_ringmo_decoder_mapper(decoder_prefix)

    def mapper(torch_key):
        result = encoder_mapper(torch_key)
        if result is not None:
            return result
        return decoder_mapper(torch_key)

    return mapper
