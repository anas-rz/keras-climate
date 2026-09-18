"""Build/shape sanity checks + PyTorch weight-port round-trip test for
`keras_climate.foundation.anysat`.

Important caveat (see `weights/mappings/anysat_mapping.py`'s module
docstring): the *officially released* AnySat checkpoint uses a
substantially more complex, config-driven architecture (per-modality
projector zoo, patch dropout, a cross-modal relative-position-encoding
block) than `AnySatEncoder` implements here. This test validates the
weight-porting *infrastructure* against a from-scratch PyTorch mirror of
this repo's own (simpler, self-consistent) design - it does not port the
real public AnySat checkpoint.

Run with: pytest keras_climate/foundation/test_anysat.py
"""
import numpy as np
import pytest
import keras

from keras_climate.foundation.anysat import AnySatEncoder, AnySatClassifier
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_anysat_mapper


# --------------------------------------------------------------------------
# Build / forward-pass sanity checks (Keras only, no torch required)
# --------------------------------------------------------------------------

def test_encoder_builds_and_runs_with_multiple_modalities():
    specs = {"s2": {"patch_size_px": 8, "gsd_m": 10.0}, "s1": {"patch_size_px": 4, "gsd_m": 20.0}}
    encoder = AnySatEncoder(specs, embed_dim=32, depth=2, num_heads=4)
    x = {"s2": np.random.randn(2, 32, 32, 4).astype("float32"),
         "s1": np.random.randn(2, 16, 16, 2).astype("float32")}
    out = encoder(x)
    num_patches = (32 // 8) ** 2 + (16 // 4) ** 2
    assert tuple(out.shape) == (2, num_patches + 1, 32)


def test_classifier():
    specs = {"s2": {"patch_size_px": 8, "gsd_m": 10.0}}
    clf = AnySatClassifier(specs, embed_dim=32, depth=2, num_heads=4,
                            input_shapes={"s2": (32, 32, 4)}, num_classes=5)
    y = keras.ops.convert_to_numpy(clf({"s2": np.random.randn(1, 32, 32, 4).astype("float32")}))
    assert y.shape == (1, 5)


def test_missing_modality_is_simply_omitted():
    """A given batch can supply only a subset of the modalities the
    encoder was configured for - this is the whole point of the design."""
    specs = {"s2": {"patch_size_px": 8, "gsd_m": 10.0}, "s1": {"patch_size_px": 4, "gsd_m": 20.0}}
    encoder = AnySatEncoder(specs, embed_dim=32, depth=1, num_heads=4)
    out_both = encoder({"s2": np.random.randn(1, 32, 32, 4).astype("float32"),
                         "s1": np.random.randn(1, 16, 16, 2).astype("float32")})
    out_s2_only = encoder({"s2": np.random.randn(1, 32, 32, 4).astype("float32")})
    assert out_both.shape[1] > out_s2_only.shape[1]


# --------------------------------------------------------------------------
# PyTorch weight-port round-trip against a from-scratch mirror of this
# repo's own AnySatEncoder design (see module docstring: this is not the
# officially released AnySat checkpoint's architecture).
# --------------------------------------------------------------------------

def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(out)

    class Mlp(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attn(dim, num_heads)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = Mlp(dim, int(dim * mlp_ratio))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class CoordMlp(nn.Module):
        def __init__(self, dim, num_freqs=8, max_freq=1024.0):
            super().__init__()
            self.num_freqs = num_freqs
            self.max_freq = max_freq
            self.fc1 = nn.Linear(4 * num_freqs, dim)
            self.fc2 = nn.Linear(dim, dim)

        def forward(self, coords):
            freqs = self.max_freq ** (torch.arange(self.num_freqs).float() / self.num_freqs)
            args = coords[..., None] / freqs
            feats = torch.cat([args.sin(), args.cos()], dim=-1)
            B, N = coords.shape[0], coords.shape[1]
            feats = feats.reshape(B, N, 4 * self.num_freqs)
            return self.fc2(torch.nn.functional.gelu(self.fc1(feats)))

    class TorchAnySatEncoder(nn.Module):
        def __init__(self, modality_specs, embed_dim, depth, num_heads):
            super().__init__()
            self.modality_specs = modality_specs
            self.embed_dim = embed_dim
            self.embeds = nn.ModuleDict({
                m: nn.Conv2d(4 if m == "s2" else 2, embed_dim, spec["patch_size_px"], spec["patch_size_px"])
                for m, spec in modality_specs.items()
            })
            self.modtoks = nn.ParameterDict({
                m: nn.Parameter(torch.zeros(1, 1, embed_dim)) for m in modality_specs
            })
            self.pos_encoding = CoordMlp(embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        def forward(self, inputs):
            all_tokens = []
            B = None
            for modality, tensor in inputs.items():
                spec = self.modality_specs[modality]
                tok = self.embeds[modality](tensor)  # (B, D, H, W)
                B, D, H, W = tok.shape
                tok = tok.flatten(2).transpose(1, 2)  # (B, HW, D)
                patch_extent = spec["patch_size_px"] * spec["gsd_m"]
                ys = (torch.arange(H).float() + 0.5) * patch_extent
                xs = (torch.arange(W).float() + 0.5) * patch_extent
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                coords = torch.stack([gx.flatten(), gy.flatten()], dim=-1)[None].expand(B, -1, -1)
                tok = tok + self.pos_encoding(coords) + self.modtoks[modality]
                all_tokens.append(tok)
            tokens = torch.cat(all_tokens, dim=1)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            for blk in self.blocks:
                tokens = blk(tokens)
            return self.norm(tokens)

    torch.manual_seed(0)
    specs = {"s2": {"patch_size_px": 8, "gsd_m": 10.0}, "s1": {"patch_size_px": 4, "gsd_m": 20.0}}
    embed_dim, depth, num_heads = 32, 2, 4

    torch_model = TorchAnySatEncoder(specs, embed_dim, depth, num_heads)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    flat = {}
    for m in specs:
        flat[f"embed_{m}.proj.weight"] = torch_model.embeds[m].weight.detach().numpy()
        flat[f"embed_{m}.proj.bias"] = torch_model.embeds[m].bias.detach().numpy()
        flat[f"modtok_{m}"] = torch_model.modtoks[m].detach().numpy()
    flat["cls_token"] = torch_model.cls_token.detach().numpy()
    flat["pos_encoding.coord_mlp.fc1.weight"] = torch_model.pos_encoding.fc1.weight.detach().numpy()
    flat["pos_encoding.coord_mlp.fc1.bias"] = torch_model.pos_encoding.fc1.bias.detach().numpy()
    flat["pos_encoding.coord_mlp.fc2.weight"] = torch_model.pos_encoding.fc2.weight.detach().numpy()
    flat["pos_encoding.coord_mlp.fc2.bias"] = torch_model.pos_encoding.fc2.bias.detach().numpy()
    for i, blk in enumerate(torch_model.blocks):
        for k, v in blk.state_dict().items():
            flat[f"blocks.{i}.{k}"] = v.detach().numpy()
    flat["norm.weight"] = torch_model.norm.weight.detach().numpy()
    flat["norm.bias"] = torch_model.norm.bias.detach().numpy()

    keras_model = AnySatEncoder(specs, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                 name="anysat_encoder")
    keras_model({"s2": np.zeros((1, 32, 32, 4), dtype="float32"),
                 "s1": np.zeros((1, 16, 16, 2), dtype="float32")})  # build

    mapper = build_anysat_mapper(list(specs), keras_prefix="anysat_encoder")
    report = WeightConverter(keras_model, flat, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_s2 = np.random.randn(1, 32, 32, 4).astype("float32")
    x_s1 = np.random.randn(1, 16, 16, 2).astype("float32")
    with torch.no_grad():
        torch_out = torch_model({
            "s2": torch.from_numpy(np.transpose(x_s2, (0, 3, 1, 2))),
            "s1": torch.from_numpy(np.transpose(x_s1, (0, 3, 1, 2))),
        }).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model({"s2": x_s2, "s1": x_s1}, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-3, f"AnySat weight port numerical mismatch: max abs diff {max_diff}"
