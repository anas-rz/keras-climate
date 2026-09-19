import re


def _attn_rules(torch_prefix, keras_prefix):
    tp = re.escape(torch_prefix)
    rules = []
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        rules += [
            (rf"^{tp}\.{name}\.weight$", f"{keras_prefix}/{name}/kernel"),
            (rf"^{tp}\.{name}\.bias$", f"{keras_prefix}/{name}/bias"),
        ]
    return rules


def _encoder_layer_rules(i):
    prefix = f"encoder_layer{i}"
    rules = _attn_rules(f"{prefix}.attn", f"{prefix}/attn")
    for n in (1, 2):
        rules += [
            (rf"^{prefix}\.norm{n}\.weight$", f"{prefix}/norm{n}/gamma"),
            (rf"^{prefix}\.norm{n}\.bias$", f"{prefix}/norm{n}/beta"),
            (rf"^{prefix}\.conv{n}\.weight$", f"{prefix}/conv{n}/kernel"),
            (rf"^{prefix}\.conv{n}\.bias$", f"{prefix}/conv{n}/bias"),
        ]
    return rules


def _decoder_layer_rules(i):
    prefix = f"decoder_layer{i}"
    rules = _attn_rules(f"{prefix}.self_attn", f"{prefix}/self_attn")
    rules += _attn_rules(f"{prefix}.cross_attn", f"{prefix}/cross_attn")
    for n in (1, 2, 3):
        if n <= 2:
            rules += [
                (rf"^{prefix}\.conv{n}\.weight$", f"{prefix}/conv{n}/kernel"),
                (rf"^{prefix}\.conv{n}\.bias$", f"{prefix}/conv{n}/bias"),
            ]
        rules += [
            (rf"^{prefix}\.norm{n}\.weight$", f"{prefix}/norm{n}/gamma"),
            (rf"^{prefix}\.norm{n}\.bias$", f"{prefix}/norm{n}/beta"),
        ]
    return rules


def _embedding_rules(torch_prefix, keras_prefix):
    tp = re.escape(torch_prefix)
    return [
        (rf"^{tp}\.value_embedding\.conv\.weight$", f"{keras_prefix}/value_embedding/conv/kernel"),
    ]


def build_informer_mapper(encoder_layers, decoder_layers):
    rules = []
    rules += _embedding_rules("enc_embedding", "enc_embedding")
    rules += _embedding_rules("dec_embedding", "dec_embedding")

    for i in range(encoder_layers):
        rules += _encoder_layer_rules(i)
        if i < encoder_layers - 1:
            rules += [
                (rf"^distill{i}\.conv\.weight$", f"distill{i}/conv/kernel"),
                (rf"^distill{i}\.bn\.weight$", f"distill{i}/bn/gamma"),
                (rf"^distill{i}\.bn\.bias$", f"distill{i}/bn/beta"),
                (rf"^distill{i}\.bn\.running_mean$", f"distill{i}/bn/moving_mean"),
                (rf"^distill{i}\.bn\.running_var$", f"distill{i}/bn/moving_variance"),
            ]

    for i in range(decoder_layers):
        rules += _decoder_layer_rules(i)

    rules += [
        (r"^projection\.weight$", "projection/kernel"),
        (r"^projection\.bias$", "projection/bias"),
    ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
