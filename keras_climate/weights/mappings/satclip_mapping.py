"""
Mapping for `keras_climate.remote_sensing.satclip.SatCLIPLocationEncoder`.

The official checkpoint is a PyTorch Lightning `.ckpt` (a dict with an
`epoch`/`state_dict`/`hyper_parameters`/... wrapper, not a plain
`state_dict` file) whose `state_dict` stores the location encoder's
SirenNet weights TWICE, under two different key prefixes that alias the
same underlying `nn.Module` (`model.nnet.*` and `model.location.nnet.*`)
- `load_satclip_checkpoint` unwraps the outer Lightning dict; the mapper
below only targets the `model.location.nnet.*` alias and the loader skips
the `model.nnet.*` duplicate (plus the image-encoder `model.visual.*`
weights and `model.logit_scale`, all out of scope - see `satclip.py`'s
module docstring for why only the location encoder is ported).
"""

import re


def load_satclip_checkpoint(path):
    """Loads a PyTorch Lightning `.ckpt` and returns its inner
    `state_dict` as `{key: np.ndarray}` (requires `torch`, used only at
    conversion time)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.detach().numpy() for k, v in ckpt["state_dict"].items()}


def build_satclip_location_mapper(num_hidden_layers=2):
    """`SatCLIPLocationEncoder` is always a top-level `keras.Model`
    (Functional API), so its own `name` never becomes a path prefix on its
    layers' weights (only nested sub-models/layers add their name as a
    path segment) - the target Keras keys here are flat, matching
    `layers{i}/kernel` etc. directly."""
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
    r"^model\.nnet\.",  # duplicate alias of model.location.nnet.* (same tensors)
]
