import re


def _corr_rules(torch_prefix, keras_prefix):
    tp = re.escape(torch_prefix)
    rules = []
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        rules += [
            (rf"^{tp}\.{name}\.weight$", f"{keras_prefix}/{name}/kernel"),
            (rf"^{tp}\.{name}\.bias$", f"{keras_prefix}/{name}/bias"),
        ]
    return rules


def _embedding_rules(torch_prefix, keras_prefix):
    tp = re.escape(torch_prefix)
    return [(rf"^{tp}\.conv\.weight$", f"{keras_prefix}/conv/kernel")]


def _encoder_layer_rules(i):
    prefix = f"encoder_layer{i}"
    rules = _corr_rules(f"{prefix}.auto_correlation", f"{prefix}/auto_correlation")
    for n in (1, 2):
        rules += [
            (rf"^{prefix}\.conv{n}\.weight$", f"{prefix}/conv{n}/kernel"),
            (rf"^{prefix}\.conv{n}\.bias$", f"{prefix}/conv{n}/bias"),
        ]
    return rules


def _decoder_layer_rules(i):
    prefix = f"decoder_layer{i}"
    rules = _corr_rules(f"{prefix}.self_correlation", f"{prefix}/self_correlation")
    rules += _corr_rules(f"{prefix}.cross_correlation", f"{prefix}/cross_correlation")
    for n in (1, 2):
        rules += [
            (rf"^{prefix}\.conv{n}\.weight$", f"{prefix}/conv{n}/kernel"),
            (rf"^{prefix}\.conv{n}\.bias$", f"{prefix}/conv{n}/bias"),
        ]
    rules += [(rf"^{prefix}\.trend_proj\.weight$", f"{prefix}/trend_proj/kernel")]
    return rules


def build_autoformer_mapper(encoder_layers, decoder_layers):
    rules = []
    rules += _embedding_rules("enc_embedding", "enc_embedding")
    rules += _embedding_rules("dec_embedding", "dec_embedding")

    for i in range(encoder_layers):
        rules += _encoder_layer_rules(i)
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
