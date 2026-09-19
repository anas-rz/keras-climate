import numpy as np
import pytest
import keras

from keras_climate.remote_sensing.scalemae import (
    ScaleMAE,
    ScaleMAEEncoder,
    GSDPositionalEmbedding,
)
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_scalemae_mapper


def test_builds_and_runs():
    model = ScaleMAE(
        img_size=64, patch_size=16, in_chans=3, embed_dim=32, depth=2, num_heads=4
    )
    x = np.random.randn(2, 64, 64, 3).astype("float32")
    res = np.array([0.3, 1.5], dtype="float32")
    tokens = keras.ops.convert_to_numpy(model([x, res]))
    num_patches = (64 // 16) ** 2
    assert tokens.shape == (2, num_patches + 1, 32)


def test_different_gsd_gives_different_positional_embedding():
    pos_embed = GSDPositionalEmbedding(grid_size=4, dim=32)
    res_a = np.array([0.3, 0.3], dtype="float32")
    res_b = np.array([0.3, 3.0], dtype="float32")
    out = keras.ops.convert_to_numpy(pos_embed(res_b))
    out_a = keras.ops.convert_to_numpy(pos_embed(res_a))
    assert np.allclose(out[0], out_a[0])
    assert not np.allclose(out[1], out_a[1])


def test_cls_token_position_is_always_zero():
    pos_embed = GSDPositionalEmbedding(grid_size=4, dim=16)
    out = keras.ops.convert_to_numpy(pos_embed(np.array([0.1, 9.9], dtype="float32")))
    assert np.allclose(out[:, 0, :], 0.0)


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
            self.scale = self.head_dim**-0.5

        def forward(self, x):
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
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

    class ViTBlock(nn.Module):
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

    def gsd_pos_embed(grid_size, dim, res):
        grid_h, grid_w = np.meshgrid(
            np.arange(grid_size, dtype=np.float32),
            np.arange(grid_size, dtype=np.float32),
            indexing="ij",
        )
        grid_h, grid_w = grid_h.reshape(-1), grid_w.reshape(-1)
        omega = 1.0 / (10000 ** (np.arange(dim // 4, dtype=np.float32) / (dim / 4.0)))

        gh = grid_h[None, :] * res[:, None]
        gw = grid_w[None, :] * res[:, None]
        out_h = gh[..., None] * omega[None, None, :]
        out_w = gw[..., None] * omega[None, None, :]
        emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=-1)
        emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=-1)
        pos = np.concatenate([emb_h, emb_w], axis=-1)
        cls_pos = np.zeros((res.shape[0], 1, dim), dtype=np.float32)
        return np.concatenate([cls_pos, pos], axis=1)

    class TorchScaleMAEEncoder(nn.Module):
        def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads):
            super().__init__()
            self.grid_size = img_size // patch_size
            self.embed_dim = embed_dim
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.blocks = nn.ModuleList(
                [ViTBlock(embed_dim, num_heads) for _ in range(depth)]
            )
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        def forward(self, x, res):
            x = self.patch_embed.proj(x)
            B = x.shape[0]
            x = x.flatten(2).transpose(1, 2)
            pos = torch.from_numpy(gsd_pos_embed(self.grid_size, self.embed_dim, res))
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1) + pos
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    torch.manual_seed(0)
    img, patch, in_chans, embed, depth, heads = 64, 16, 3, 32, 2, 4

    torch_model = TorchScaleMAEEncoder(img, patch, in_chans, embed, depth, heads)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    encoder = ScaleMAEEncoder(
        img_size=img,
        patch_size=patch,
        in_chans=in_chans,
        embed_dim=embed,
        depth=depth,
        num_heads=heads,
    )
    encoder(
        [
            np.zeros((1, img, img, in_chans), dtype="float32"),
            np.zeros((1,), dtype="float32"),
        ]
    )

    mapper = build_scalemae_mapper()
    report = WeightConverter(encoder, state_dict, mapper).convert(
        strict=False, verbose=False
    )
    assert set(report["missing_in_source"]) == {
        "scalemae_encoder/pos_embed/grid_h",
        "scalemae_encoder/pos_embed/grid_w",
        "scalemae_encoder/pos_embed/omega",
    }
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, in_chans, img, img).astype("float32")
    res_np = np.array([0.3, 1.7], dtype="float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np), res_np).numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    keras_out = keras.ops.convert_to_numpy(encoder([keras_in, res_np], training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert (
        max_diff < 1e-2
    ), f"ScaleMAE weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_scalemae_vitlarge_fmow():
    pytest.importorskip("torch")
    from keras_climate.weights.pretrained import scalemae_vitlarge_fmow

    encoder, report = scalemae_vitlarge_fmow(img_size=224)
    assert set(report["missing_in_source"]) == {
        "scalemae_encoder/pos_embed/grid_h",
        "scalemae_encoder/pos_embed/grid_w",
        "scalemae_encoder/pos_embed/omega",
    }
    assert not report["unused_source_keys"]

    x = np.random.rand(1, 224, 224, 3).astype("float32")
    res = np.array([0.6], dtype="float32")
    tokens = keras.ops.convert_to_numpy(encoder([x, res]))
    assert tokens.shape == (1, (224 // 16) ** 2 + 1, 1024)
    assert np.isfinite(tokens).all()
