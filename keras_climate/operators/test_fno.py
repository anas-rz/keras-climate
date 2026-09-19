import numpy as np
import pytest
import keras

from keras_climate.operators.fno import FNO2D, SpectralConv2D
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_fno_mapper


def test_builds_and_runs():
    model = FNO2D(input_shape=(32, 32, 1), out_channels=1, width=16, modes1=6, modes2=6,
                   num_layers=2)
    x = np.random.randn(2, 32, 32, 1).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 32, 32, 1)


def test_resolution_invariance():
    model = FNO2D(input_shape=(16, 16, 1), out_channels=1, width=8, modes1=4, modes2=4,
                   num_layers=1, add_grid=False)
    x_small = np.random.randn(1, 16, 16, 1).astype("float32")
    y_small = keras.ops.convert_to_numpy(model(x_small))
    assert y_small.shape == (1, 16, 16, 1)

    model2 = FNO2D(input_shape=(32, 32, 1), out_channels=1, width=8, modes1=4, modes2=4,
                    num_layers=1, add_grid=False)
    model2.set_weights(model.get_weights())
    x_large = np.random.randn(1, 32, 32, 1).astype("float32")
    y_large = keras.ops.convert_to_numpy(model2(x_large))
    assert y_large.shape == (1, 32, 32, 1)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchSpectralConv2d(nn.Module):
        def __init__(self, in_c, out_c, modes1, modes2):
            super().__init__()
            self.in_c, self.out_c, self.modes1, self.modes2 = in_c, out_c, modes1, modes2
            scale = 1 / (in_c * out_c)
            self.weights1 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, dtype=torch.cfloat))

        def compl_mul2d(self, x, w):
            return torch.einsum("bixy,ioxy->boxy", x, w)

        def forward(self, x):
            B, C, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="backward")
            out_ft = torch.zeros(B, self.out_c, H, W // 2 + 1, dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            return torch.fft.irfft2(out_ft, s=(H, W), norm="backward")

    class TorchBlock(nn.Module):
        def __init__(self, width, modes1, modes2, activation):
            super().__init__()
            self.spectral = TorchSpectralConv2d(width, width, modes1, modes2)
            self.pointwise = nn.Conv2d(width, width, 1)
            self.activation = activation

        def forward(self, x):
            out = self.spectral(x) + self.pointwise(x)
            return F.gelu(out) if self.activation else out

    class TorchFNO2D(nn.Module):
        def __init__(self, in_chans, out_chans, width, modes1, modes2, num_layers, H, W):
            super().__init__()
            self.fc0 = nn.Linear(in_chans + 2, width)
            self.blocks = nn.ModuleList(
                [TorchBlock(width, modes1, modes2, i < num_layers - 1) for i in range(num_layers)])
            self.fc1 = nn.Linear(width, 128)
            self.fc2 = nn.Linear(128, out_chans)
            gx, gy = np.meshgrid(np.linspace(0, 1, H, dtype="float32"),
                                  np.linspace(0, 1, W, dtype="float32"), indexing="ij")
            self.register_buffer("grid", torch.from_numpy(np.stack([gx, gy], axis=-1)))

        def forward(self, x):
            B = x.shape[0]
            grid = self.grid.unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, grid], dim=-1)
            x = self.fc0(x)
            x = x.permute(0, 3, 1, 2)
            for blk in self.blocks:
                x = blk(x)
            x = x.permute(0, 2, 3, 1)
            x = F.gelu(self.fc1(x))
            return self.fc2(x)

    torch.manual_seed(0)
    H, W, in_chans, out_chans = 16, 16, 1, 1
    width, modes1, modes2, num_layers = 8, 4, 4, 2

    torch_model = TorchFNO2D(in_chans, out_chans, width, modes1, modes2, num_layers, H, W)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            if p.is_complex():
                p.real.normal_(0.0, 0.05)
                p.imag.normal_(0.0, 0.05)
            else:
                p.normal_(0.0, 0.2)

    state_dict = {}
    for k, v in torch_model.state_dict().items():
        if k == "grid":
            continue
        if v.is_complex():
            base = k.rsplit(".", 1)[0]
            name = k.rsplit(".", 1)[1]
            state_dict[f"{base}.{name}_re"] = v.real.numpy()
            state_dict[f"{base}.{name}_im"] = v.imag.numpy()
        else:
            state_dict[k] = v.detach().numpy()

    keras_model = FNO2D(input_shape=(H, W, in_chans), out_channels=out_chans, width=width,
                         modes1=modes1, modes2=modes2, num_layers=num_layers, add_grid=True)

    mapper = build_fno_mapper(num_layers=num_layers)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=False, verbose=False)
    assert report["missing_in_source"] == ["coord_grid/grid"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, H, W, in_chans).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model(x_np, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"FNO weight port numerical mismatch: max abs diff {max_diff}"
