"""Build/shape sanity checks + PyTorch weight-port round-trip test for
`keras_climate.forecasting.dlinear`.

Run with: pytest keras_climate/forecasting/test_dlinear.py
"""
import numpy as np
import pytest
import keras

from keras_climate.forecasting.dlinear import DLinear, SeriesDecomposition
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_dlinear_mapper


# --------------------------------------------------------------------------
# Build / forward-pass sanity checks (Keras only, no torch required)
# --------------------------------------------------------------------------

def test_builds_and_runs_shared():
    model = DLinear(seq_len=96, pred_len=48, num_channels=7, individual=False)
    x = np.random.randn(2, 96, 7).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 48, 7)


def test_builds_and_runs_individual():
    model = DLinear(seq_len=96, pred_len=48, num_channels=3, individual=True)
    x = np.random.randn(2, 96, 3).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 48, 3)


def test_decomposition_seasonal_plus_trend_equals_input():
    decomp = SeriesDecomposition(kernel_size=5)
    x = np.random.randn(2, 20, 3).astype("float32")
    seasonal, trend = decomp(x)
    recon = keras.ops.convert_to_numpy(seasonal + trend)
    assert np.allclose(recon, x, atol=1e-5)


# --------------------------------------------------------------------------
# PyTorch weight-port round-trip against a from-scratch reference matching
# the official Zeng et al. DLinear repo's own naming.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("individual", [False, True])
def test_weight_port_roundtrip_matches_pytorch_reference(individual):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchSeriesDecomp(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

        def forward(self, x):
            # x: (B, L, C)
            pad_front = (self.kernel_size - 1) // 2
            pad_end = self.kernel_size - 1 - pad_front
            front = x[:, :1, :].repeat(1, pad_front, 1)
            end = x[:, -1:, :].repeat(1, pad_end, 1)
            padded = torch.cat([front, x, end], dim=1)
            trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
            return x - trend, trend

    class TorchDLinear(nn.Module):
        def __init__(self, seq_len, pred_len, num_channels, kernel_size, individual):
            super().__init__()
            self.decomp = TorchSeriesDecomp(kernel_size)
            self.individual = individual
            self.num_channels = num_channels
            if individual:
                self.Linear_Seasonal = nn.ModuleList(
                    [nn.Linear(seq_len, pred_len) for _ in range(num_channels)])
                self.Linear_Trend = nn.ModuleList(
                    [nn.Linear(seq_len, pred_len) for _ in range(num_channels)])
            else:
                self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
                self.Linear_Trend = nn.Linear(seq_len, pred_len)

        def forward(self, x):
            seasonal, trend = self.decomp(x)
            seasonal, trend = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)  # (B, C, L)
            if self.individual:
                s_out = torch.stack([self.Linear_Seasonal[c](seasonal[:, c, :])
                                      for c in range(self.num_channels)], dim=1)
                t_out = torch.stack([self.Linear_Trend[c](trend[:, c, :])
                                      for c in range(self.num_channels)], dim=1)
            else:
                s_out = self.Linear_Seasonal(seasonal)
                t_out = self.Linear_Trend(trend)
            return (s_out + t_out).permute(0, 2, 1)  # (B, pred_len, C)

    torch.manual_seed(0)
    seq_len, pred_len, num_channels, kernel_size = 32, 16, 3, 5

    torch_model = TorchDLinear(seq_len, pred_len, num_channels, kernel_size, individual)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = DLinear(seq_len=seq_len, pred_len=pred_len, num_channels=num_channels,
                           kernel_size=kernel_size, individual=individual)

    mapper = build_dlinear_mapper(num_channels=num_channels, individual=individual)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, seq_len, num_channels).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model(x_np, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"DLinear weight port numerical mismatch: max abs diff {max_diff}"
