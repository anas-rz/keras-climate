import numpy as np
import pytest
import keras

from keras_climate.weather.earthformer import (
    Earthformer,
    CuboidAttention,
    CuboidTransformerBlock,
)
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_earthformer_mapper


@pytest.mark.parametrize("strategy", ["local", "dilated"])
def test_cuboid_partition_reassemble_is_exact_inverse(strategy):
    attn = CuboidAttention(dim=4, num_heads=2, cuboid_size=(2, 4, 4), strategy=strategy)
    attn.build((1, 4, 8, 8, 4))
    x = np.random.randn(1, 4, 8, 8, 4).astype("float32")
    cuboids, meta = attn._partition(x, 4, 8, 8)
    recon = keras.ops.convert_to_numpy(attn._reassemble(cuboids, meta))
    assert np.allclose(recon, x)


def test_builds_and_runs():
    model = Earthformer(
        input_shape=(4, 32, 32, 1),
        pred_steps=4,
        base_dim=8,
        stage_depths=(1, 1),
        num_heads=2,
        cuboid_size=(2, 4, 4),
    )
    x = np.random.randn(1, 4, 32, 32, 1).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (1, 4, 32, 32, 1)


def test_different_pred_steps_and_channels():
    model = Earthformer(
        input_shape=(2, 16, 16, 2),
        pred_steps=6,
        base_dim=8,
        stage_depths=(1,),
        num_heads=2,
        cuboid_size=(2, 4, 4),
    )
    x = np.random.randn(1, 2, 16, 16, 2).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (1, 6, 16, 16, 2)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    def partition(x, ct, ch, cw, strategy):
        B, T, H, W, C = x.shape
        nt, nh, nw = T // ct, H // ch, W // cw
        if strategy == "local":
            x = x.reshape(B, nt, ct, nh, ch, nw, cw, C)
            x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        else:
            x = x.reshape(B, ct, nt, ch, nh, cw, nw, C)
            x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.reshape(B * nt * nh * nw, ct * ch * cw, C), (
            B,
            nt,
            nh,
            nw,
            ct,
            ch,
            cw,
            C,
        )

    def reassemble(x, meta, strategy):
        B, nt, nh, nw, ct, ch, cw, C = meta
        x = x.reshape(B, nt, nh, nw, ct, ch, cw, C)
        if strategy == "local":
            x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        else:
            x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(B, nt * ct, nh * ch, nw * cw, C)

    class Attn(nn.Module):
        def __init__(self, dim, num_heads, cuboid_size, strategy="local"):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim**-0.5
            self.cuboid_size = cuboid_size
            self.strategy = strategy
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, T, H, W, C = x.shape
            ct, ch, cw = self.cuboid_size
            cuboids, meta = partition(x, ct, ch, cw, self.strategy)
            N = cuboids.shape[1]
            qkv = self.qkv(cuboids).reshape(
                cuboids.shape[0], N, 3, self.num_heads, self.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(cuboids.shape[0], N, self.dim)
            out = self.proj(out)
            return reassemble(out, meta, self.strategy)

    class Mlp(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class Block(nn.Module):
        def __init__(
            self, dim, num_heads, cuboid_size, strategy="local", mlp_ratio=4.0
        ):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attn(dim, num_heads, cuboid_size, strategy)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = Mlp(dim, int(dim * mlp_ratio))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class Stage(nn.Module):
        def __init__(self, dim, num_heads, depth, cuboid_size):
            super().__init__()
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim,
                        num_heads,
                        cuboid_size,
                        strategy="local" if i % 2 == 0 else "dilated",
                    )
                    for i in range(depth)
                ]
            )

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

    class TorchEarthformer(nn.Module):

        def __init__(
            self, T_in, H, W, C_in, pred_steps, base_dim, depth, num_heads, cuboid_size
        ):
            super().__init__()
            self.stem = nn.Conv2d(C_in, base_dim, 3, padding=1)
            self.enc_stage0 = Stage(base_dim, num_heads, depth, cuboid_size)
            self.time_channel_proj = nn.Linear(T_in * base_dim, pred_steps * C_in)
            self.T_in, self.H, self.W, self.C_in = T_in, H, W, C_in
            self.pred_steps, self.base_dim = pred_steps, base_dim

        def forward(self, x):
            B, T, C, H, W = x.shape
            x = self.stem(x.reshape(B * T, C, H, W)).reshape(B, T, self.base_dim, H, W)
            x = x.permute(0, 1, 3, 4, 2)
            x = self.enc_stage0(x)
            x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, T * self.base_dim)
            x = self.time_channel_proj(x)
            x = x.reshape(B, H, W, self.pred_steps, self.C_in)
            return x.permute(0, 3, 1, 2, 4)

    torch.manual_seed(0)
    T_in, H, W, C_in = 2, 16, 16, 1
    pred_steps, base_dim, depth, num_heads, cuboid_size = 3, 8, 2, 2, (2, 4, 4)

    torch_model = TorchEarthformer(
        T_in, H, W, C_in, pred_steps, base_dim, depth, num_heads, cuboid_size
    )
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    flat = {}
    flat["stem.conv.weight"] = torch_model.stem.weight.detach().numpy()
    flat["stem.conv.bias"] = torch_model.stem.bias.detach().numpy()
    for i, blk in enumerate(torch_model.enc_stage0.blocks):
        p = f"enc_stage0.blocks.{i}"
        flat[f"{p}.norm1.weight"] = blk.norm1.weight.detach().numpy()
        flat[f"{p}.norm1.bias"] = blk.norm1.bias.detach().numpy()
        flat[f"{p}.attn.qkv.weight"] = blk.attn.qkv.weight.detach().numpy()
        flat[f"{p}.attn.qkv.bias"] = blk.attn.qkv.bias.detach().numpy()
        flat[f"{p}.attn.proj.weight"] = blk.attn.proj.weight.detach().numpy()
        flat[f"{p}.attn.proj.bias"] = blk.attn.proj.bias.detach().numpy()
        flat[f"{p}.norm2.weight"] = blk.norm2.weight.detach().numpy()
        flat[f"{p}.norm2.bias"] = blk.norm2.bias.detach().numpy()
        flat[f"{p}.mlp.fc1.weight"] = blk.mlp.fc1.weight.detach().numpy()
        flat[f"{p}.mlp.fc1.bias"] = blk.mlp.fc1.bias.detach().numpy()
        flat[f"{p}.mlp.fc2.weight"] = blk.mlp.fc2.weight.detach().numpy()
        flat[f"{p}.mlp.fc2.bias"] = blk.mlp.fc2.bias.detach().numpy()
    flat["time_channel_proj.weight"] = (
        torch_model.time_channel_proj.weight.detach().numpy()
    )
    flat["time_channel_proj.bias"] = torch_model.time_channel_proj.bias.detach().numpy()

    keras_model = Earthformer(
        input_shape=(T_in, H, W, C_in),
        pred_steps=pred_steps,
        base_dim=base_dim,
        stage_depths=(depth,),
        num_heads=num_heads,
        cuboid_size=cuboid_size,
    )
    mapper = build_earthformer_mapper(
        num_enc_stages=1, num_dec_stages=0, enc_depths=[depth]
    )
    report = WeightConverter(keras_model, flat, mapper).convert(
        strict=True, verbose=False
    )
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(1, T_in, C_in, H, W).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_in = np.transpose(x_np, (0, 1, 3, 4, 2))
    keras_out = keras.ops.convert_to_numpy(keras_model(keras_in, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert (
        max_diff < 1e-3
    ), f"Earthformer weight port numerical mismatch: max abs diff {max_diff}"
