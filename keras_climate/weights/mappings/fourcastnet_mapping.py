"""
Mapping for `keras_climate.weather.fourcastnet.FourCastNet`, targeting
NVIDIA's official checkpoint (mirrored at NERSC - see
`weights/pretrained.py`'s `fourcastnet_backbone` loader).

Two real (not cosmetic) checkpoint quirks, both handled by
`convert_fourcastnet_state_dict` rather than the generic `WeightConverter`
shape-inference machinery (which only knows well-known transpose
patterns, not a DDP prefix strip or an arbitrary reshape):
  * every key is nested under a `module.` prefix (the checkpoint was
    saved from inside `torch.nn.parallel.DistributedDataParallel`).
  * `pos_embed` is stored flattened as `(1, grid_h*grid_w, C)`, whereas
    `LearnedPositionEmbedding2D` stores it as `(1, grid_h, grid_w, C)` to
    match this module's `(B, H, W, C)` token-grid layout throughout -
    reshaping (not transposing) it into that shape is a one-off, so it's
    done directly rather than via a `param_kind`.

Everything else (`patch_embed`, `blocks.{i}.*`, `norm`, `head`) reuses
`build_afno_mapper` (shared with `keras_climate.operators.afno`), plus two
extra rules here for `pos_embed`/`norm` (not part of the generic AFNO
block naming). `head` has no bias in the official checkpoint (`nn.Linear
(..., bias=False)`), matching `FourCastNet`'s own `use_bias=False` head.
"""

import re

from keras_climate.weights.mappings.afno_mapping import build_afno_mapper


def load_fourcastnet_checkpoint(path):
    """Loads NVIDIA's official FourCastNet training checkpoint (a dict
    with `iters`/`epoch`/`model_state`/`optimizer_state_dict` keys, saved
    from inside `DistributedDataParallel`) and returns its `model_state`
    as `{key: np.ndarray}`. Requires `torch` (conversion-time only); also
    requires `ruamel.yaml` installed, since the pickled checkpoint embeds
    a config object of that type even though only the plain tensor
    `model_state` dict is actually used here."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.detach().numpy() for k, v in ckpt["model_state"].items()}


def convert_fourcastnet_state_dict(flat_state_dict, grid_h, grid_w):
    """Strips the `module.` DDP-wrapper prefix and reshapes `pos_embed`
    from `(1, grid_h*grid_w, C)` to `(1, grid_h, grid_w, C)`."""
    out = {}
    for k, v in flat_state_dict.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k == "pos_embed":
            v = v.reshape(1, grid_h, grid_w, v.shape[-1])
        out[k] = v
    return out


def build_fourcastnet_mapper(depth):
    afno_mapper = build_afno_mapper(depth)
    rules = [
        (r"^pos_embed$", "pos_embed_layer/pos_embed"),
        (r"^norm\.weight$", "norm/gamma"),
        (r"^norm\.bias$", "norm/beta"),
    ]
    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return afno_mapper(torch_key)

    return mapper
