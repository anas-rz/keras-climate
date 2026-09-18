"""
Mapping for `keras_climate.operators.deeponet.DeepONet`, assuming a
source checkpoint using this repo's own naming (no verifiable real
checkpoint exists for this exact architecture - see `deeponet.py`'s
module docstring):

    branch_fc{i}.{weight,bias}
    trunk_fc{i}.{weight,bias}
    bias                                   (raw scalar parameter)
"""

import re


def build_deeponet_mapper(num_branch_layers, num_trunk_layers):
    rules = [(r"^bias$", "bias_layer/bias")]
    for i in range(num_branch_layers):
        rules += [
            (rf"^branch_fc{i}\.weight$", f"branch_fc{i}/kernel"),
            (rf"^branch_fc{i}\.bias$", f"branch_fc{i}/bias"),
        ]
    for i in range(num_trunk_layers):
        rules += [
            (rf"^trunk_fc{i}\.weight$", f"trunk_fc{i}/kernel"),
            (rf"^trunk_fc{i}\.bias$", f"trunk_fc{i}/bias"),
        ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
