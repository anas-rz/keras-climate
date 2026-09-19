import numpy as np
import pytest
import keras

from keras_climate.forecasting.patchtst import PatchTST, RevIN
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_patchtst_mapper


def test_builds_and_runs():
    model = PatchTST(
        seq_len=96,
        pred_len=24,
        num_channels=3,
        patch_len=16,
        stride=8,
        d_model=32,
        depth=2,
        num_heads=4,
    )
    x = np.random.randn(2, 96, 3).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 24, 3)


def test_end_padding_adds_one_patch():
    common = dict(
        seq_len=96,
        pred_len=24,
        num_channels=3,
        patch_len=16,
        stride=8,
        d_model=16,
        depth=1,
        num_heads=2,
    )
    padded = PatchTST(**common, padding_patch="end")
    unpadded = PatchTST(**common, padding_patch=None)
    padded_patches = padded.get_layer("pos_embed").num_patches
    unpadded_patches = unpadded.get_layer("pos_embed").num_patches
    assert padded_patches == unpadded_patches + 1


def test_revin_weights_are_tracked_and_trainable():
    model = PatchTST(
        seq_len=48,
        pred_len=12,
        num_channels=2,
        patch_len=8,
        stride=4,
        d_model=16,
        depth=1,
        num_heads=2,
    )
    revin_weight_names = [w.path for w in model.weights if "revin" in w.path]
    assert set(revin_weight_names) == {"revin/gamma", "revin/beta"}
    assert model.get_layer("revin") in model.layers


def test_positional_embedding_is_tracked_and_trainable():
    model = PatchTST(
        seq_len=48,
        pred_len=12,
        num_channels=2,
        patch_len=8,
        stride=4,
        d_model=16,
        depth=1,
        num_heads=2,
    )
    assert model.get_layer("pos_embed") in model.layers
    assert any(w.path == "pos_embed/embeddings" for w in model.weights)


def test_revin_denormalize_inverts_normalize():
    revin = RevIN(affine=False)
    x = (np.random.randn(4, 20, 3) * 10 + 5).astype("float32")
    normed = revin(x, mode="norm")
    denormed = keras.ops.convert_to_numpy(revin(normed, mode="denorm"))
    assert np.allclose(denormed, x, atol=1e-4)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3)
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

    class Mlp(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=2.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attn(dim, num_heads)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = Mlp(dim, int(dim * mlp_ratio))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class TorchRevIN(nn.Module):
        def __init__(self, num_features, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        def norm(self, x):
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (
                (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            )
            x = (x - self.mean) / self.stdev
            return x * self.affine_weight + self.affine_bias

        def denorm(self, x):
            x = (x - self.affine_bias) / (self.affine_weight + self.eps**2)
            return x * self.stdev + self.mean

    class TorchPatchTST(nn.Module):
        def __init__(
            self,
            seq_len,
            pred_len,
            num_channels,
            patch_len,
            stride,
            d_model,
            depth,
            num_heads,
        ):
            super().__init__()
            self.revin = TorchRevIN(num_channels)
            self.patch_len, self.stride = patch_len, stride
            self.num_channels = num_channels
            padded_len = seq_len + stride
            self.num_patches = (padded_len - patch_len) // stride + 1
            self.patch_proj = nn.Linear(patch_len, d_model)
            self.pos_embed = nn.Parameter(torch.zeros(self.num_patches, d_model))
            self.blocks = nn.ModuleList(
                [Block(d_model, num_heads) for _ in range(depth)]
            )
            self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)
            self.forecast_head = nn.Linear(self.num_patches * d_model, pred_len)
            self.pred_len = pred_len

        def forward(self, x):
            x = self.revin.norm(x)
            x = x.transpose(1, 2)
            last = x[:, :, -1:].expand(-1, -1, self.stride)
            x = torch.cat([x, last], dim=-1)
            patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            B, C, P, PL = patches.shape
            patches = patches.reshape(B * C, P, PL)
            tokens = self.patch_proj(patches) + self.pos_embed[None]
            for blk in self.blocks:
                tokens = blk(tokens)
            tokens = self.encoder_norm(tokens)
            flat = tokens.reshape(B * C, -1)
            head = self.forecast_head(flat)
            out = head.reshape(B, C, self.pred_len).transpose(1, 2)
            return self.revin.denorm(out)

    torch.manual_seed(0)
    seq_len, pred_len, num_channels, patch_len, stride = 32, 8, 2, 8, 4
    d_model, depth, num_heads = 16, 1, 2

    torch_model = TorchPatchTST(
        seq_len, pred_len, num_channels, patch_len, stride, d_model, depth, num_heads
    )
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = PatchTST(
        seq_len=seq_len,
        pred_len=pred_len,
        num_channels=num_channels,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        depth=depth,
        num_heads=num_heads,
        dropout=0.0,
        padding_patch="end",
    )

    mapper = build_patchtst_mapper(depth=depth)
    report = WeightConverter(keras_model, state_dict, mapper).convert(
        strict=True, verbose=False
    )
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, seq_len, num_channels).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model(x_np, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert (
        max_diff < 1e-2
    ), f"PatchTST weight port numerical mismatch: max abs diff {max_diff}"
