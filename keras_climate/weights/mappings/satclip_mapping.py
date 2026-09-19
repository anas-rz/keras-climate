import re


def load_satclip_checkpoint(path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.detach().numpy() for k, v in ckpt["state_dict"].items()}


def build_satclip_location_mapper(num_hidden_layers=2):
    rules = []
    for i in range(num_hidden_layers):
        rules += [
            (rf"^model\.location\.nnet\.layers\.{i}\.weight$", f"layers{i}/kernel"),
            (rf"^model\.location\.nnet\.layers\.{i}\.bias$", f"layers{i}/bias"),
        ]
    rules += [
        (r"^model\.location\.nnet\.last_layer\.weight$", "last_layer/kernel"),
        (r"^model\.location\.nnet\.last_layer\.bias$", "last_layer/bias"),
    ]
    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper


SATCLIP_SKIP_PATTERNS = [
    r"^model\.visual\.",
    r"^model\.logit_scale$",
    r"^model\.posenc",
    r"^model\.location\.posenc",
    r"^model\.nnet\.",
]
