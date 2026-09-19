import numpy as np
import pytest
import keras

from keras_climate.operators.afno import AFNOOperator, AFNO2D
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_afno_mapper


def test_builds_and_runs():
    model = AFNOOperator(
        input_shape=(32, 32, 2),
        out_channels=2,
        patch_size=4,
        embed_dim=16,
        depth=2,
        num_blocks=4,
    )
    x = np.random.randn(2, 32, 32, 2).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 32, 32, 2)


def test_afno2d_preserves_grid_shape():
    layer = AFNO2D(hidden_size=16, num_blocks=4)
    x = np.random.randn(2, 8, 8, 16).astype("float32")
    y = keras.ops.convert_to_numpy(layer(x))
    assert y.shape == (2, 8, 8, 16)


def test_afno2d_is_residual_when_filter_weights_are_zero():
    layer = AFNO2D(hidden_size=8, num_blocks=2)
    x = np.random.randn(2, 4, 4, 8).astype("float32")
    layer(x)
    layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])
    y = keras.ops.convert_to_numpy(layer(x))
    assert np.allclose(y, x, atol=1e-4)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchAFNO2D(nn.Module):
        def __init__(
            self, hidden_size, num_blocks, sparsity_threshold=0.01, hidden_size_factor=1
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_blocks = num_blocks
            self.block_size = hidden_size // num_blocks
            self.sparsity_threshold = sparsity_threshold
            scale = 0.02
            bs, hf, nb = self.block_size, hidden_size_factor, num_blocks
            self.w1 = nn.Parameter(scale * torch.randn(2, nb, bs, bs * hf))
            self.b1 = nn.Parameter(scale * torch.randn(2, nb, bs * hf))
            self.w2 = nn.Parameter(scale * torch.randn(2, nb, bs * hf, bs))
            self.b2 = nn.Parameter(scale * torch.randn(2, nb, bs))

        def forward(self, x):
            bias = x
            B, H, W, C = x.shape
            x = torch.fft.rfft2(x, dim=(1, 2), norm="backward")
            x_re, x_im = x.real, x.imag
            x_re = x_re.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
            x_im = x_im.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

            o1_re = F.relu(
                torch.einsum("bhwnc,ncd->bhwnd", x_re, self.w1[0])
                - torch.einsum("bhwnc,ncd->bhwnd", x_im, self.w1[1])
                + self.b1[0]
            )
            o1_im = F.relu(
                torch.einsum("bhwnc,ncd->bhwnd", x_im, self.w1[0])
                + torch.einsum("bhwnc,ncd->bhwnd", x_re, self.w1[1])
                + self.b1[1]
            )
            o2_re = (
                torch.einsum("bhwnd,nde->bhwne", o1_re, self.w2[0])
                - torch.einsum("bhwnd,nde->bhwne", o1_im, self.w2[1])
                + self.b2[0]
            )
            o2_im = (
                torch.einsum("bhwnd,nde->bhwne", o1_im, self.w2[0])
                + torch.einsum("bhwnd,nde->bhwne", o1_re, self.w2[1])
                + self.b2[1]
            )
            o2_re = o2_re.reshape(B, H, W // 2 + 1, C)
            o2_im = o2_im.reshape(B, H, W // 2 + 1, C)

            stacked = torch.stack([o2_re, o2_im], dim=-1)
            stacked = F.softshrink(stacked, lambd=self.sparsity_threshold)
            out = torch.view_as_complex(stacked)
            out = torch.fft.irfft2(out, s=(H, W), dim=(1, 2), norm="backward")
            return out + bias

    torch.manual_seed(0)
    hidden_size, num_blocks, H, W = 16, 4, 8, 8

    torch_layer = TorchAFNO2D(hidden_size, num_blocks)
    torch_layer.eval()

    keras_layer = AFNO2D(hidden_size, num_blocks, name="filter")
    keras_layer(np.zeros((1, H, W, hidden_size), dtype="float32"))

    state_dict = {k: v.detach().numpy() for k, v in torch_layer.state_dict().items()}
    mapper = {
        "w1": "filter/w1",
        "b1": "filter/b1",
        "w2": "filter/w2",
        "b2": "filter/b2",
    }
    report = WeightConverter(keras_layer, state_dict, lambda k: mapper.get(k)).convert(
        strict=True, verbose=False
    )
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, H, W, hidden_size).astype("float32")
    with torch.no_grad():
        torch_out = torch_layer(torch.from_numpy(x_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_layer(x_np))

    max_diff = np.abs(torch_out - keras_out).max()
    assert (
        max_diff < 1e-2
    ), f"AFNO2D weight port numerical mismatch: max abs diff {max_diff}"


def test_afno_operator_mapper_matches_full_model():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    depth, embed_dim, num_blocks, patch_size = 2, 16, 4, 4
    input_shape = (16, 16, 2)

    class TorchPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(
                input_shape[-1], embed_dim, patch_size, stride=patch_size
            )

    class TorchDummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = TorchPatchEmbed()
            self.blocks = nn.ModuleList()
            for _ in range(depth):
                blk = nn.Module()
                blk.norm1 = nn.LayerNorm(embed_dim)
                blk.filter = nn.Module()
                bs = embed_dim // num_blocks
                blk.filter.w1 = nn.Parameter(torch.zeros(2, num_blocks, bs, bs))
                blk.filter.b1 = nn.Parameter(torch.zeros(2, num_blocks, bs))
                blk.filter.w2 = nn.Parameter(torch.zeros(2, num_blocks, bs, bs))
                blk.filter.b2 = nn.Parameter(torch.zeros(2, num_blocks, bs))
                blk.norm2 = nn.LayerNorm(embed_dim)
                blk.mlp = nn.Module()
                blk.mlp.fc1 = nn.Linear(embed_dim, embed_dim * 4)
                blk.mlp.fc2 = nn.Linear(embed_dim * 4, embed_dim)
                self.blocks.append(blk)

    torch_model = TorchDummy()
    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = AFNOOperator(
        input_shape=input_shape,
        out_channels=2,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_blocks=num_blocks,
    )

    mapper = build_afno_mapper(depth=depth)
    report = WeightConverter(keras_model, state_dict, mapper).convert(
        strict=False, verbose=False
    )
    assert all(k.startswith("head/") for k in report["missing_in_source"])
    assert not report["unused_source_keys"]
