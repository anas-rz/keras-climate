"""
Mapping for `keras_climate.forecasting.nbeats.NBeats`, assuming a source
checkpoint using this repo's own naming (no official/general-purpose
checkpoint exists - see `nbeats.py`'s module docstring):

    block{i}.fc{j}.{weight,bias}          (j in [0, num_fc_layers))
    block{i}.theta.{weight,bias}

Trend/Seasonality basis buffers (`backcast_basis`, `forecast_basis`,
`{backcast,forecast}_{cos,sin}`) are derived purely from config (degree/
num_harmonics/backcast_size/forecast_size), not learned - they have no
source-key counterpart and should be excluded from a strict conversion
check (see `test_nbeats.py`).
"""

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
