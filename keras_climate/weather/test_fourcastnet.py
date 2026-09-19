import numpy as np
import pytest
import keras

from keras_climate.weather.fourcastnet import FourCastNet
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_fourcastnet_mapper, convert_fourcastnet_state_dict


def test_builds_and_runs():
    model = FourCastNet(img_size=(32, 32), patch_size=4, in_chans=5, out_chans=5,
                         embed_dim=16, depth=2, num_blocks=4)
    x = np.random.randn(2, 32, 32, 5).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 32, 32, 5)


def test_non_square_grid():
    model = FourCastNet(img_size=(16, 32), patch_size=4, in_chans=3, out_chans=3,
                         embed_dim=8, depth=1, num_blocks=2)
    x = np.random.randn(1, 16, 32, 3).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (1, 16, 32, 3)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchAFNO2D(nn.Module):
        def __init__(self, hidden_size, num_blocks, sparsity_threshold=0.01):
            super().__init__()
            self.num_blocks = num_blocks
            self.block_size = hidden_size // num_blocks
            self.sparsity_threshold = sparsity_threshold
            scale = 0.02
            bs, nb = self.block_size, num_blocks
            self.w1 = nn.Parameter(scale * torch.randn(2, nb, bs, bs))
            self.b1 = nn.Parameter(scale * torch.randn(2, nb, bs))
            self.w2 = nn.Parameter(scale * torch.randn(2, nb, bs, bs))
            self.b2 = nn.Parameter(scale * torch.randn(2, nb, bs))

        def forward(self, x):
            bias = x
            B, H, W, C = x.shape
            xf = torch.fft.rfft2(x, dim=(1, 2), norm="backward")
            x_re, x_im = xf.real, xf.imag
            x_re = x_re.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
            x_im = x_im.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

            o1_re = F.relu(torch.einsum("bhwnc,ncd->bhwnd", x_re, self.w1[0])
                            - torch.einsum("bhwnc,ncd->bhwnd", x_im, self.w1[1]) + self.b1[0])
            o1_im = F.relu(torch.einsum("bhwnc,ncd->bhwnd", x_im, self.w1[0])
                            + torch.einsum("bhwnc,ncd->bhwnd", x_re, self.w1[1]) + self.b1[1])
            o2_re = (torch.einsum("bhwnd,nde->bhwne", o1_re, self.w2[0])
                      - torch.einsum("bhwnd,nde->bhwne", o1_im, self.w2[1]) + self.b2[0])
            o2_im = (torch.einsum("bhwnd,nde->bhwne", o1_im, self.w2[0])
                      + torch.einsum("bhwnd,nde->bhwne", o1_re, self.w2[1]) + self.b2[1])
            o2_re, o2_im = o2_re.reshape(B, H, W // 2 + 1, C), o2_im.reshape(B, H, W // 2 + 1, C)

            stacked = F.softshrink(torch.stack([o2_re, o2_im], dim=-1), lambd=self.sparsity_threshold)
            out = torch.view_as_complex(stacked)
            out = torch.fft.irfft2(out, s=(H, W), dim=(1, 2), norm="backward")
            return out + bias

    class TorchMlp(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))

    class TorchAFNOBlock(nn.Module):
        def __init__(self, dim, num_blocks, mlp_ratio):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.filter = TorchAFNO2D(dim, num_blocks)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = TorchMlp(dim, int(dim * mlp_ratio))

        def forward(self, x):
            x = x + self.filter(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class TorchFourCastNet(nn.Module):
        def __init__(self, H, W, patch_size, in_chans, out_chans, embed_dim, depth,
                     num_blocks, mlp_ratio):
            super().__init__()
            self.patch_size = patch_size
            self.grid_h, self.grid_w = H // patch_size, W // patch_size
            self.out_chans = out_chans
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_h * self.grid_w, embed_dim))
            self.blocks = nn.ModuleList(
                [TorchAFNOBlock(embed_dim, num_blocks, mlp_ratio) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
            self.head = nn.Linear(embed_dim, patch_size * patch_size * out_chans, bias=False)

        def forward(self, x):
            x = self.patch_embed.proj(x).permute(0, 2, 3, 1)
            x = x + self.pos_embed.reshape(1, self.grid_h, self.grid_w, -1)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            x = self.head(x)
            B = x.shape[0]
            p, oc = self.patch_size, self.out_chans
            x = x.reshape(B, self.grid_h, self.grid_w, p, p, oc)
            x = x.permute(0, 1, 3, 2, 4, 5)
            return x.reshape(B, self.grid_h * p, self.grid_w * p, oc)

    torch.manual_seed(0)
    H, W, patch_size, in_chans, out_chans = 16, 16, 4, 3, 3
    embed_dim, depth, num_blocks, mlp_ratio = 16, 2, 4, 4.0

    torch_model = TorchFourCastNet(H, W, patch_size, in_chans, out_chans, embed_dim, depth,
                                    num_blocks, mlp_ratio)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.2)

    raw_state_dict = {f"module.{k}": v.detach().numpy() for k, v in torch_model.state_dict().items()}
    grid_h, grid_w = H // patch_size, W // patch_size
    state_dict = convert_fourcastnet_state_dict(raw_state_dict, grid_h, grid_w)

    keras_model = FourCastNet(img_size=(H, W), patch_size=patch_size, in_chans=in_chans,
                               out_chans=out_chans, embed_dim=embed_dim, depth=depth,
                               num_blocks=num_blocks, mlp_ratio=mlp_ratio)

    mapper = build_fourcastnet_mapper(depth=depth)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, in_chans, H, W).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    keras_out = keras.ops.convert_to_numpy(keras_model(keras_in, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"FourCastNet weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_fourcastnet_backbone():
    pytest.importorskip("torch")
    from keras_climate.weights.pretrained import fourcastnet_backbone

    model, report = fourcastnet_backbone()
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x = np.random.rand(1, 720, 1440, 20).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (1, 720, 1440, 20)
    assert np.isfinite(y).all()
