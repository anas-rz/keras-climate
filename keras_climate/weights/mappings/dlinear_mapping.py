"""
Mapping for `keras_climate.forecasting.dlinear.DLinear`, assuming a source
checkpoint using the official Zeng et al. DLinear repo's own naming:

    Linear_Seasonal.{weight,bias}                  (individual=False)
    Linear_Trend.{weight,bias}
    Linear_Seasonal.{c}.{weight,bias}              (individual=True)
    Linear_Trend.{c}.{weight,bias}

No decomposition-layer weights exist to map (moving-average pooling has
no learnable parameters).
"""

import re


def build_dlinear_mapper(num_channels=7, individual=False):
    rules = []
    if individual:
        for c in range(num_channels):
            rules += [
                (rf"^Linear_Seasonal\.{c}\.weight$", f"linear_seasonal{c}/kernel"),
                (rf"^Linear_Seasonal\.{c}\.bias$", f"linear_seasonal{c}/bias"),
                (rf"^Linear_Trend\.{c}\.weight$", f"linear_trend{c}/kernel"),
                (rf"^Linear_Trend\.{c}\.bias$", f"linear_trend{c}/bias"),
            ]
    else:
        rules += [
            (r"^Linear_Seasonal\.weight$", "linear_seasonal/kernel"),
            (r"^Linear_Seasonal\.bias$", "linear_seasonal/bias"),
            (r"^Linear_Trend\.weight$", "linear_trend/kernel"),
            (r"^Linear_Trend\.bias$", "linear_trend/bias"),
        ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
