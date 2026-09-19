import re


def build_fno_mapper(num_layers):
    rules = [
        (r"^fc0\.weight$", "fc0/kernel"),
        (r"^fc0\.bias$", "fc0/bias"),
        (r"^fc1\.weight$", "fc1/kernel"),
        (r"^fc1\.bias$", "fc1/bias"),
        (r"^fc2\.weight$", "fc2/kernel"),
        (r"^fc2\.bias$", "fc2/bias"),
    ]
    for i in range(num_layers):
        prefix = f"blocks.{i}"
        kp = f"blocks{i}"
        for w in ("weights1_re", "weights1_im", "weights2_re", "weights2_im"):
            rules.append((rf"^{prefix}\.spectral\.{w}$", f"{kp}/spectral/{w}"))
        rules += [
            (rf"^{prefix}\.pointwise\.weight$", f"{kp}/pointwise/kernel"),
            (rf"^{prefix}\.pointwise\.bias$", f"{kp}/pointwise/bias"),
        ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
