import re


def _uno_block_rules(torch_prefix, keras_prefix):
    rules = [(rf"^{torch_prefix}\.spectral\.{w}$", f"{keras_prefix}/spectral/{w}")
             for w in ("weights1_re", "weights1_im", "weights2_re", "weights2_im")]
    rules += [
        (rf"^{torch_prefix}\.pointwise\.weight$", f"{keras_prefix}/pointwise/kernel"),
        (rf"^{torch_prefix}\.pointwise\.bias$", f"{keras_prefix}/pointwise/bias"),
    ]
    return rules


def build_uno_mapper(depth):
    rules = [
        (r"^lift\.weight$", "lift/kernel"),
        (r"^lift\.bias$", "lift/bias"),
        (r"^project\.weight$", "project/kernel"),
        (r"^project\.bias$", "project/bias"),
    ]
    for i in range(depth):
        rules += _uno_block_rules(f"enc_block{i}", f"enc_block{i}")
        rules += _uno_block_rules(f"dec_block{i}", f"dec_block{i}")
    rules += _uno_block_rules("bottleneck", "bottleneck")

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
