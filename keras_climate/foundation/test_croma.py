"""Build/shape sanity checks + PyTorch weight-port round-trip test for
`keras_climate.foundation.croma`.

Run with: pytest keras_climate/foundation/test_croma.py
"""
import numpy as np
import pytest
import keras

from keras_climate.foundation.croma import CROMA, get_2d_alibi
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import load_croma_checkpoint, convert_croma_state_dict, build_croma_identity_mapper


# --------------------------------------------------------------------------
# Build / forward-pass sanity checks (Keras only, no torch required)
# --------------------------------------------------------------------------

def test_alibi_bias_shape_and_symmetry():
    bias = get_2d_alibi(num_heads=4, grid_size=5)
    assert bias.shape == (1, 4, 25, 25)
    # distance-based bias must be symmetric and zero on the diagonal.
    assert np.allclose(bias[0, 0], bias[0, 0].T)
    assert np.allclose(np.diagonal(bias[0, 0]), 0.0)


def test_builds_and_runs():
    model = CROMA(img_size=32, patch_size=8, size="base")
    sar = np.random.randn(1, 32, 32, 2).astype("float32")
    opt = np.random.randn(1, 32, 32, 12).astype("float32")
    out = model([sar, opt])
    num_patches = (32 // 8) ** 2
    assert tuple(out["sar_repr"].shape) == (1, 768)
    assert tuple(out["optical_repr"].shape) == (1, 768)
    assert tuple(out["joint_tokens"].shape) == (1, num_patches, 768)


def test_sar_encoder_is_half_depth_of_optical():
    model = CROMA(img_size=32, patch_size=8, size="base")
    sar_blocks = [l for l in model.get_layer("s1_encoder").blocks]
    opt_blocks = [l for l in model.get_layer("s2_encoder").blocks]
    assert len(sar_blocks) == len(opt_blocks) // 2


# --------------------------------------------------------------------------
# PyTorch weight-port round-trip against the official architecture (see
# `foundation/croma.py`'s module docstring for why this differs so much
# from a "reasonable-sounding" ViT default: Linear-on-flattened-patches,
# 2D ALiBi instead of learned position embeddings, asymmetric SAR/optical
# depths, single-query-stream cross-attention fusion).
# --------------------------------------------------------------------------

def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from einops import rearrange, einsum

    dim, depth, num_heads, patch_size, img_size = 32, 4, 4, 8, 32
    grid = img_size // patch_size

    class FFN(nn.Module):
        def __init__(self, dim, mult=4):
            super().__init__()
            self.input_norm = nn.LayerNorm(dim)
            # Dropout(0) is a real (if inert) member of the reference's
            # `net` Sequential - keeping it preserves the index alignment
            # the mapper expects (`net.0` = fc1, `net.3` = fc2).
            self.net = nn.Sequential(nn.Linear(dim, int(dim * mult)), nn.GELU(), nn.Dropout(0.0),
                                      nn.Linear(int(dim * mult), dim))

        def forward(self, x):
            return self.net(self.input_norm(x))

    class Attention(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            dim_head = dim // num_heads
            self.scale = dim_head ** -0.5
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_out = nn.Linear(dim, dim)
            self.input_norm = nn.LayerNorm(dim)

        def forward(self, x, bias):
            x = self.input_norm(x)
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.num_heads) for t in (q, k, v))
            attn = (einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale + bias).softmax(dim=-1)
            out = einsum(attn, v, "b h i j, b h j d -> b h i d")
            return self.to_out(rearrange(out, "b h n d -> b n (h d)"))

    class CrossAttention(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            dim_head = dim // num_heads
            self.scale = dim_head ** -0.5
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.Linear(dim, dim)
            self.input_norm = nn.LayerNorm(dim)

        def forward(self, x, context, bias):
            xn, cn = self.input_norm(x), self.input_norm(context)
            q = rearrange(self.to_q(xn), "b n (h d) -> b h n d", h=self.num_heads)
            k = rearrange(self.to_k(cn), "b n (h d) -> b h n d", h=self.num_heads)
            v = rearrange(self.to_v(cn), "b n (h d) -> b h n d", h=self.num_heads)
            attn = (einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale + bias).softmax(dim=-1)
            out = einsum(attn, v, "b h i j, b h j d -> b h i d")
            return self.to_out(rearrange(out, "b h n d -> b n (h d)"))

    class BaseTransformer(nn.Module):
        def __init__(self, dim, depth, num_heads):
            super().__init__()
            self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, num_heads), FFN(dim)])
                                          for _ in range(depth)])
            self.norm_out = nn.LayerNorm(dim)

        def forward(self, x, bias):
            for attn, ffn in self.layers:
                x = attn(x, bias) + x
                x = ffn(x) + x
            return self.norm_out(x)

    class BaseTransformerCrossAttn(nn.Module):
        def __init__(self, dim, depth, num_heads):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleList([Attention(dim, num_heads), CrossAttention(dim, num_heads), FFN(dim)])
                for _ in range(depth)])
            self.norm_out = nn.LayerNorm(dim)

        def forward(self, x, context, bias):
            for self_attn, cross_attn, ffn in self.layers:
                x = self_attn(x, bias) + x
                x = cross_attn(x, context, bias) + x
                x = ffn(x) + x
            return self.norm_out(x)

    class ViT(nn.Module):
        def __init__(self, dim, depth, in_channels, patch_size, num_heads):
            super().__init__()
            self.patch_size = patch_size
            self.linear_input = nn.Linear(patch_size * patch_size * in_channels, dim)
            self.transformer = BaseTransformer(dim, depth, num_heads)

        def forward(self, imgs, bias):
            x = rearrange(imgs, "b c (h i) (w j) -> b (h w) (c i j)", i=self.patch_size, j=self.patch_size)
            return self.transformer(self.linear_input(x), bias)

    class TorchCROMA(nn.Module):
        def __init__(self, dim, depth, num_heads, patch_size, sar_chans=2, opt_chans=12):
            super().__init__()
            self.s1_encoder = ViT(dim, depth // 2, sar_chans, patch_size, num_heads)
            self.s2_encoder = ViT(dim, depth, opt_chans, patch_size, num_heads)
            self.s1_GAP_FFN = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(),
                                             nn.Linear(4 * dim, dim))
            self.s2_GAP_FFN = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(),
                                             nn.Linear(4 * dim, dim))
            self.joint_encoder = BaseTransformerCrossAttn(dim, depth // 2, num_heads)

        def forward(self, sar, opt, bias):
            sar_tok = self.s1_encoder(sar, bias)
            opt_tok = self.s2_encoder(opt, bias)
            sar_repr = self.s1_GAP_FFN(sar_tok.mean(dim=1))
            opt_repr = self.s2_GAP_FFN(opt_tok.mean(dim=1))
            joint = self.joint_encoder(sar_tok, opt_tok, bias)
            return sar_repr, opt_repr, joint

    from .croma import get_2d_alibi

    torch.manual_seed(0)
    torch_model = TorchCROMA(dim, depth, num_heads, patch_size)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    flat = {}
    for name, submodule in [("s1_encoder", torch_model.s1_encoder), ("s2_encoder", torch_model.s2_encoder)]:
        for k, v in submodule.state_dict().items():
            flat[f"{name}.{k}"] = v.detach().numpy()
    for name, submodule in [("s1_GAP_FFN", torch_model.s1_GAP_FFN), ("s2_GAP_FFN", torch_model.s2_GAP_FFN),
                             ("joint_encoder", torch_model.joint_encoder)]:
        for k, v in submodule.state_dict().items():
            flat[f"{name}.{k}"] = v.detach().numpy()

    translated = convert_croma_state_dict(flat, encoder_depth=depth)

    # `CROMA(...)` only builds the two hardcoded (dim, depth) presets
    # ("base"/"large"); assemble a small model with the same test dims
    # directly from its building blocks instead.
    from .croma import ModalityEncoder, CrossAttentionFusion, gap_ffn
    sar_in = keras.Input((img_size, img_size, 2), name="sar")
    opt_in = keras.Input((img_size, img_size, 12), name="optical")
    sar_encoder = ModalityEncoder(dim, depth // 2, 2, patch_size, num_heads, name="s1_encoder")
    opt_encoder = ModalityEncoder(dim, depth, 12, patch_size, num_heads, name="s2_encoder")
    attn_bias = keras.ops.convert_to_tensor(get_2d_alibi(num_heads, grid))
    sar_tokens = sar_encoder(sar_in, attn_bias)
    opt_tokens = opt_encoder(opt_in, attn_bias)
    fusion = CrossAttentionFusion(dim, num_heads, depth // 2, name="cross_encoder")
    joint_tokens = fusion(sar_tokens, opt_tokens, attn_bias)
    sar_pool = keras.layers.GlobalAveragePooling1D(name="sar_pool")(sar_tokens)
    opt_pool = keras.layers.GlobalAveragePooling1D(name="optical_pool")(opt_tokens)
    sar_repr = gap_ffn(dim, name="GAP_FFN_s1")(sar_pool)
    opt_repr = gap_ffn(dim, name="GAP_FFN_s2")(opt_pool)
    keras_model = keras.Model([sar_in, opt_in],
                               {"sar_repr": sar_repr, "optical_repr": opt_repr, "joint_tokens": joint_tokens})

    report = WeightConverter(keras_model, translated, build_croma_identity_mapper()).convert(
        strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_sar = np.random.randn(1, 2, img_size, img_size).astype("float32")
    x_opt = np.random.randn(1, 12, img_size, img_size).astype("float32")
    bias_t = torch.from_numpy(get_2d_alibi(num_heads, grid))
    with torch.no_grad():
        torch_sar_repr, torch_opt_repr, torch_joint = torch_model(
            torch.from_numpy(x_sar), torch.from_numpy(x_opt), bias_t)

    keras_sar = np.transpose(x_sar, (0, 2, 3, 1))
    keras_opt = np.transpose(x_opt, (0, 2, 3, 1))
    out = keras_model([keras_sar, keras_opt])

    d_sar = np.abs(torch_sar_repr.numpy() - keras.ops.convert_to_numpy(out["sar_repr"])).max()
    d_opt = np.abs(torch_opt_repr.numpy() - keras.ops.convert_to_numpy(out["optical_repr"])).max()
    d_joint = np.abs(torch_joint.numpy() - keras.ops.convert_to_numpy(out["joint_tokens"])).max()
    assert max(d_sar, d_opt, d_joint) < 1e-3, f"CROMA weight port numerical mismatch: {d_sar}, {d_opt}, {d_joint}"


@pytest.mark.pretrained
def test_real_pretrained_croma_base_checkpoint():
    """Downloads the real `antofuller/CROMA` base checkpoint and confirms
    it loads cleanly. Run explicitly with `pytest -m pretrained` (network +
    ~740MB download, cached after the first run)."""
    pytest.importorskip("torch")
    from ..weights.pretrained import croma_base

    model, report = croma_base()
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    sar = np.random.rand(1, 120, 120, 2).astype("float32")
    opt = np.random.rand(1, 120, 120, 12).astype("float32")
    out = model([sar, opt])
    assert keras.ops.convert_to_numpy(out["sar_repr"]).shape == (1, 768)
    assert np.isfinite(keras.ops.convert_to_numpy(out["joint_tokens"])).all()
