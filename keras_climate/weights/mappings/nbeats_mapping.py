import re


def _block_rules(block_idx, num_fc_layers):
    prefix = f"block{block_idx}"
    rules = []
    for j in range(num_fc_layers):
        rules += [
            (rf"^{prefix}\.fc{j}\.weight$", f"{prefix}/fc{j}/kernel"),
            (rf"^{prefix}\.fc{j}\.bias$", f"{prefix}/fc{j}/bias"),
        ]
    rules += [
        (rf"^{prefix}\.theta\.weight$", f"{prefix}/theta/kernel"),
        (rf"^{prefix}\.theta\.bias$", f"{prefix}/theta/bias"),
    ]
    return rules


def build_nbeats_mapper(num_blocks, num_fc_layers=4):
    rules = []
    for i in range(num_blocks):
        rules += _block_rules(i, num_fc_layers)

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
