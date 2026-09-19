import numpy as np
import pytest
import keras

from keras_climate.weather.climax import (
    ClimaX,
    MultiVariablePatchEmbed,
    VariableAggregation,
)
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_climax_mapper


def test_builds_and_runs():
    model = ClimaX(
        img_size=(16, 32),
        patch_size=4,
        num_vars=3,
        embed_dim=16,
        depth=2,
        decoder_depth=1,
        num_heads=4,
    )
    fields = np.random.randn(2, 16, 32, 3).astype("float32")
    lead_time = np.array([[1.0], [2.0]], dtype="float32")
    y = keras.ops.convert_to_numpy(model([fields, lead_time]))
    assert y.shape == (2, 16, 32, 3)


def test_multi_variable_patch_embed_uses_separate_weights_per_variable():
    layer = MultiVariablePatchEmbed(num_vars=3, patch_size=4, embed_dim=8)
    x = np.random.randn(1, 16, 16, 3).astype("float32")
    layer(x)
    kernels = [c.kernel.numpy() for c in layer.token_embeds]
    assert not np.allclose(kernels[0], kernels[1])


def test_variable_aggregation_output_shape():
    layer = VariableAggregation(embed_dim=8, num_heads=2)
    x = np.random.randn(2, 4, 5, 8).astype("float32")
    out = keras.ops.convert_to_numpy(layer(x))
    assert out.shape == (2, 5, 8)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchPatchEmbed(nn.Module):
        def __init__(self, patch_size, embed_dim):
            super().__init__()
            self.proj = nn.Conv2d(1, embed_dim, patch_size, stride=patch_size)

        def forward(self, x):
            x = self.proj(x)
            return x.flatten(2).transpose(1, 2)

    class TorchAttn(nn.Module):
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
            attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(out)

    class TorchMlp(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))

    class TorchBlock(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = TorchAttn(dim, num_heads)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = TorchMlp(dim, int(dim * mlp_ratio))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class TorchClimaX(nn.Module):
        def __init__(
            self,
            img_size,
            patch_size,
            num_vars,
            embed_dim,
            depth,
            decoder_depth,
            num_heads,
            mlp_ratio,
        ):
            super().__init__()
            H, W = img_size
            grid_h, grid_w = H // patch_size, W // patch_size
            num_patches = grid_h * grid_w
            self.grid_h, self.grid_w, self.patch_size, self.num_vars = (
                grid_h,
                grid_w,
                patch_size,
                num_vars,
            )

            self.token_embeds = nn.ModuleList(
                [TorchPatchEmbed(patch_size, embed_dim) for _ in range(num_vars)]
            )
            self.channel_embed = nn.Parameter(torch.zeros(1, num_vars, embed_dim))
            self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.channel_agg = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.lead_time_embed = nn.Linear(1, embed_dim)
            self.blocks = nn.ModuleList(
                [TorchBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
            )
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

            head = []
            for _ in range(decoder_depth):
                head += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
            head += [nn.Linear(embed_dim, num_vars * patch_size**2)]
            self.head = nn.Sequential(*head)

        def forward(self, fields, lead_time):
            embeds = [
                self.token_embeds[v](fields[:, v : v + 1]) for v in range(self.num_vars)
            ]
            x = torch.stack(embeds, dim=1)
            x = x + self.channel_embed.unsqueeze(2)

            b, _, l, _ = x.shape
            x = torch.einsum("bvld->blvd", x).flatten(0, 1)
            query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
            x, _ = self.channel_agg(query, x, x)
            x = x.squeeze(1).unflatten(0, (b, l))

            x = x + self.pos_embed
            lead_embed = self.lead_time_embed(lead_time)
            x = x + lead_embed.unsqueeze(1)

            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            x = self.head(x)

            B = x.shape[0]
            p, V = self.patch_size, self.num_vars
            x = x.reshape(B, self.grid_h, self.grid_w, p, p, V)
            x = x.permute(0, 1, 3, 2, 4, 5)
            return x.reshape(B, self.grid_h * p, self.grid_w * p, V)

    torch.manual_seed(0)
    img_size, patch_size, num_vars = (16, 32), 4, 3
    embed_dim, depth, decoder_depth, num_heads, mlp_ratio = 16, 2, 1, 4, 4.0

    torch_model = TorchClimaX(
        img_size,
        patch_size,
        num_vars,
        embed_dim,
        depth,
        decoder_depth,
        num_heads,
        mlp_ratio,
    )
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.2)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = ClimaX(
        img_size=img_size,
        patch_size=patch_size,
        num_vars=num_vars,
        embed_dim=embed_dim,
        depth=depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )

    mapper = build_climax_mapper(
        num_vars=num_vars, depth=depth, decoder_depth=decoder_depth
    )
    report = WeightConverter(keras_model, state_dict, mapper).convert(
        strict=True, verbose=False
    )
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    fields_np = np.random.randn(2, num_vars, *img_size).astype("float32")
    lead_time_np = np.array([[1.0], [3.0]], dtype="float32")
    with torch.no_grad():
        torch_out = torch_model(
            torch.from_numpy(fields_np), torch.from_numpy(lead_time_np)
        ).numpy()

    fields_keras = np.transpose(fields_np, (0, 2, 3, 1))
    keras_out = keras.ops.convert_to_numpy(
        keras_model([fields_keras, lead_time_np], training=False)
    )

    max_diff = np.abs(torch_out - keras_out).max()
    assert (
        max_diff < 1e-2
    ), f"ClimaX weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_climax_1_40625deg():
    pytest.importorskip("torch")
    from keras_climate.weights.pretrained import climax_1_40625deg

    model, report = climax_1_40625deg()
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    fields = np.random.rand(1, 128, 256, 48).astype("float32")
    lead_time = np.array([[1.0]], dtype="float32")
    y = keras.ops.convert_to_numpy(model([fields, lead_time]))
    assert y.shape == (1, 128, 256, 48)
    assert np.isfinite(y).all()
