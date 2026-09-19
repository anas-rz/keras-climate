import re

from keras_climate.weights.mappings.afno_mapping import build_afno_mapper


def load_fourcastnet_checkpoint(path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.detach().numpy() for k, v in ckpt["model_state"].items()}


def convert_fourcastnet_state_dict(flat_state_dict, grid_h, grid_w):
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
