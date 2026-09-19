import numpy as np
import pytest
import keras

from keras_climate.operators.uno import UNO
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_uno_mapper


def test_builds_and_runs():
    model = UNO(input_shape=(32, 32, 1), out_channels=1, base_width=4, depth=2, modes1=2, modes2=2)
    x = np.random.randn(2, 32, 32, 1).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 32, 32, 1)


def test_depth_one():
    model = UNO(input_shape=(16, 16, 2), out_channels=3, base_width=4, depth=1, modes1=2, modes2=2)
    x = np.random.randn(1, 16, 16, 2).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (1, 16, 16, 3)


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

    class TorchUNOBlock(nn.Module):
        def __init__(self, in_c, out_c, modes1, modes2):
            super().__init__()
            self.spectral = TorchSpectralConv2d(in_c, out_c, modes1, modes2)
            self.pointwise = nn.Conv2d(in_c, out_c, 1)

        def forward(self, x):
            x_nchw = x.permute(0, 3, 1, 2)
            out = self.spectral(x_nchw) + self.pointwise(x_nchw)
            return F.gelu(out.permute(0, 2, 3, 1))

    class TorchUNO(nn.Module):
        def __init__(self, in_chans, out_chans, base_width, depth, modes1, modes2):
            super().__init__()
            self.depth = depth
            self.lift = nn.Linear(in_chans, base_width)
            widths = [base_width * (2 ** i) for i in range(depth + 1)]
            prev = base_width
            for i in range(depth):
                setattr(self, f"enc_block{i}", TorchUNOBlock(prev, widths[i], modes1, modes2))
                prev = widths[i]
            self.bottleneck = TorchUNOBlock(prev, widths[depth], modes1, modes2)
            prev = widths[depth]
            for i in reversed(range(depth)):
                setattr(self, f"dec_block{i}", TorchUNOBlock(prev + widths[i], widths[i], modes1, modes2))
                prev = widths[i]
            self.project = nn.Linear(prev, out_chans)

        def forward(self, x):
            x = self.lift(x)
            skips = []
            for i in range(self.depth):
                x = getattr(self, f"enc_block{i}")(x)
                skips.append(x)
                x = F.avg_pool2d(x.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
            x = self.bottleneck(x)
            for i in reversed(range(self.depth)):
                x = F.interpolate(x.permute(0, 3, 1, 2), scale_factor=2, mode="bilinear",
                                   align_corners=False).permute(0, 2, 3, 1)
                x = torch.cat([x, skips[i]], dim=-1)
                x = getattr(self, f"dec_block{i}")(x)
            return self.project(x)

    torch.manual_seed(0)
    in_chans, out_chans, base_width, depth, modes1, modes2 = 1, 1, 4, 2, 2, 2
    H = W = 32

    torch_model = TorchUNO(in_chans, out_chans, base_width, depth, modes1, modes2)
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
        if v.is_complex():
            base, name = k.rsplit(".", 1)
            state_dict[f"{base}.{name}_re"] = v.real.numpy()
            state_dict[f"{base}.{name}_im"] = v.imag.numpy()
        else:
            state_dict[k] = v.detach().numpy()

    keras_model = UNO(input_shape=(H, W, in_chans), out_channels=out_chans, base_width=base_width,
                       depth=depth, modes1=modes1, modes2=modes2)

    mapper = build_uno_mapper(depth=depth)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, H, W, in_chans).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model(x_np, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"UNO weight port numerical mismatch: max abs diff {max_diff}"
