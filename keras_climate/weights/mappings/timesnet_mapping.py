import re


def build_timesnet_mapper(num_layers, num_kernels, use_revin=True):
    rules = [
        (r"^value_embed\.weight$", "value_embed/kernel"),
        (r"^value_embed\.bias$", "value_embed/bias"),
        (r"^predict_linear\.weight$", "predict_linear/kernel"),
        (r"^predict_linear\.bias$", "predict_linear/bias"),
        (r"^output_proj\.weight$", "output_proj/kernel"),
        (r"^output_proj\.bias$", "output_proj/bias"),
    ]
    if use_revin:
        rules += [
            (r"^revin\.affine_weight$", "revin/gamma"),
            (r"^revin\.affine_bias$", "revin/beta"),
        ]

    for i in range(num_layers):
        rules += [
            (rf"^norm{i}\.weight$", f"norm{i}/gamma"),
            (rf"^norm{i}\.bias$", f"norm{i}/beta"),
        ]
        for conv_name in ("conv1", "conv2"):
            for k in range(num_kernels):
                tp = f"timesblock{i}.{conv_name}.convs.{k}"
                kp = f"timesblock{i}/{conv_name}/kernel{k}"
                rules += [
                    (rf"^{re.escape(tp)}\.weight$", f"{kp}/kernel"),
                    (rf"^{re.escape(tp)}\.bias$", f"{kp}/bias"),
                ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
