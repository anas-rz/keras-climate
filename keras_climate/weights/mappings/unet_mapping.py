import re


def _double_conv_rules(torch_prefix, keras_prefix):
    tp = re.escape(torch_prefix)
    return [
        (rf"^{tp}\.double_conv\.0\.weight$", f"{keras_prefix}/conv1/conv/kernel"),
        (rf"^{tp}\.double_conv\.1\.weight$", f"{keras_prefix}/conv1/bn/gamma"),
        (rf"^{tp}\.double_conv\.1\.bias$", f"{keras_prefix}/conv1/bn/beta"),
        (
            rf"^{tp}\.double_conv\.1\.running_mean$",
            f"{keras_prefix}/conv1/bn/moving_mean",
        ),
        (
            rf"^{tp}\.double_conv\.1\.running_var$",
            f"{keras_prefix}/conv1/bn/moving_variance",
        ),
        (rf"^{tp}\.double_conv\.3\.weight$", f"{keras_prefix}/conv2/conv/kernel"),
        (rf"^{tp}\.double_conv\.4\.weight$", f"{keras_prefix}/conv2/bn/gamma"),
        (rf"^{tp}\.double_conv\.4\.bias$", f"{keras_prefix}/conv2/bn/beta"),
        (
            rf"^{tp}\.double_conv\.4\.running_mean$",
            f"{keras_prefix}/conv2/bn/moving_mean",
        ),
        (
            rf"^{tp}\.double_conv\.4\.running_var$",
            f"{keras_prefix}/conv2/bn/moving_variance",
        ),
    ]


def build_mapper(depth=4):
    rules = []
    rules += _double_conv_rules("inc", "enc0")
    for i in range(1, depth):
        rules += _double_conv_rules(f"down{i}.maxpool_conv.1", f"enc{i}")
    rules += _double_conv_rules(f"down{depth}.maxpool_conv.1", "bottleneck")

    for k in range(1, depth + 1):
        dec_idx = depth - k
        rules += [
            (rf"^up{k}\.up\.weight$", f"upconv{dec_idx}/kernel"),
            (rf"^up{k}\.up\.bias$", f"upconv{dec_idx}/bias"),
        ]
        rules += _double_conv_rules(f"up{k}.conv", f"dec{dec_idx}")

    rules += [
        (r"^outc\.conv\.weight$", "logits/kernel"),
        (r"^outc\.conv\.bias$", "logits/bias"),
    ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
