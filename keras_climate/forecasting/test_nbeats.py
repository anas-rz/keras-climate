import numpy as np
import pytest
import keras

from keras_climate.forecasting.nbeats import (
    NBeats,
    GenericBasis,
    TrendBasis,
    SeasonalityBasis,
)
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_nbeats_mapper


@pytest.mark.parametrize(
    "stack_types", [("generic",), ("trend", "seasonality"), ("generic", "trend")]
)
def test_builds_and_runs(stack_types):
    model = NBeats(
        seq_len=48,
        pred_len=12,
        stack_types=stack_types,
        num_blocks_per_stack=2,
        hidden_dim=32,
        num_fc_layers=2,
        trend_degree=2,
        num_harmonics=3,
    )
    x = np.random.randn(4, 48).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (4, 12)


def test_generic_basis_splits_theta_directly():
    basis = GenericBasis(backcast_size=10, forecast_size=5)
    theta = np.arange(15, dtype="float32")[None]
    backcast, forecast = basis(theta)
    assert np.allclose(keras.ops.convert_to_numpy(backcast)[0], np.arange(10))
    assert np.allclose(keras.ops.convert_to_numpy(forecast)[0], np.arange(10, 15))


def test_trend_basis_degree_zero_is_constant():
    basis = TrendBasis(degree=0, backcast_size=8, forecast_size=4)
    theta = np.array([[2.0, 3.0]], dtype="float32")
    backcast, forecast = basis(theta)
    backcast = keras.ops.convert_to_numpy(backcast)[0]
    forecast = keras.ops.convert_to_numpy(forecast)[0]
    assert np.allclose(forecast, 2.0)
    assert np.allclose(backcast, 3.0)


def test_seasonality_basis_is_periodic_over_forecast_horizon():
    basis = SeasonalityBasis(num_harmonics=1, backcast_size=8, forecast_size=8)
    theta = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    _, forecast = basis(theta)
    forecast = keras.ops.convert_to_numpy(forecast)[0]
    expected = np.cos(2 * np.pi * np.arange(8) / 8)
    assert np.allclose(forecast, expected, atol=1e-5)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchGenericBasis(nn.Module):
        def __init__(self, backcast_size, forecast_size):
            super().__init__()
            self.backcast_size = backcast_size

        def forward(self, theta):
            return theta[:, : self.backcast_size], theta[:, self.backcast_size :]

    class TorchTrendBasis(nn.Module):
        def __init__(self, degree, backcast_size, forecast_size):
            super().__init__()
            self.degree = degree
            poly = degree + 1
            self.register_buffer(
                "backcast_basis",
                torch.from_numpy(
                    np.stack(
                        [
                            (np.arange(backcast_size) / backcast_size) ** i
                            for i in range(poly)
                        ]
                    ).astype("float32")
                ),
            )
            self.register_buffer(
                "forecast_basis",
                torch.from_numpy(
                    np.stack(
                        [
                            (np.arange(forecast_size) / forecast_size) ** i
                            for i in range(poly)
                        ]
                    ).astype("float32")
                ),
            )

        def forward(self, theta):
            poly = self.degree + 1
            forecast_theta, backcast_theta = theta[:, :poly], theta[:, poly:]
            return (
                backcast_theta @ self.backcast_basis,
                forecast_theta @ self.forecast_basis,
            )

    class TorchSeasonalityBasis(nn.Module):
        def __init__(self, num_harmonics, backcast_size, forecast_size):
            super().__init__()
            self.H = num_harmonics
            k = np.arange(1, num_harmonics + 1)[:, None]
            t_back = np.arange(backcast_size)[None, :] / forecast_size
            t_fore = np.arange(forecast_size)[None, :] / forecast_size
            self.register_buffer(
                "backcast_cos",
                torch.from_numpy(np.cos(2 * np.pi * k * t_back).astype("float32")),
            )
            self.register_buffer(
                "backcast_sin",
                torch.from_numpy(np.sin(2 * np.pi * k * t_back).astype("float32")),
            )
            self.register_buffer(
                "forecast_cos",
                torch.from_numpy(np.cos(2 * np.pi * k * t_fore).astype("float32")),
            )
            self.register_buffer(
                "forecast_sin",
                torch.from_numpy(np.sin(2 * np.pi * k * t_fore).astype("float32")),
            )

        def forward(self, theta):
            H = self.H
            fc, fs = theta[:, :H], theta[:, H : 2 * H]
            bc, bs = theta[:, 2 * H : 3 * H], theta[:, 3 * H : 4 * H]
            forecast = fc @ self.forecast_cos + fs @ self.forecast_sin
            backcast = bc @ self.backcast_cos + bs @ self.backcast_sin
            return backcast, forecast

    class TorchBlock(nn.Module):
        def __init__(self, in_dim, hidden_dim, num_fc_layers, basis, theta_size):
            super().__init__()
            dims = [in_dim] + [hidden_dim] * num_fc_layers
            for i in range(num_fc_layers):
                setattr(self, f"fc{i}", nn.Linear(dims[i], dims[i + 1]))
            self.num_fc_layers = num_fc_layers
            self.theta = nn.Linear(hidden_dim, theta_size)
            self.basis = basis

        def forward(self, x):
            h = x
            for i in range(self.num_fc_layers):
                h = F.relu(getattr(self, f"fc{i}")(h))
            return self.basis(self.theta(h))

    def make_basis(stack_type, seq_len, pred_len, degree, harmonics):
        if stack_type == "generic":
            return TorchGenericBasis(seq_len, pred_len), seq_len + pred_len
        if stack_type == "trend":
            return TorchTrendBasis(degree, seq_len, pred_len), 2 * (degree + 1)
        return TorchSeasonalityBasis(harmonics, seq_len, pred_len), 4 * harmonics

    class TorchNBeats(nn.Module):
        def __init__(
            self,
            seq_len,
            pred_len,
            stack_types,
            num_blocks_per_stack,
            hidden_dim,
            num_fc_layers,
            trend_degree,
            num_harmonics,
        ):
            super().__init__()
            blocks = []
            for stack_type in stack_types:
                for _ in range(num_blocks_per_stack):
                    basis, theta_size = make_basis(
                        stack_type, seq_len, pred_len, trend_degree, num_harmonics
                    )
                    blocks.append(
                        TorchBlock(
                            seq_len, hidden_dim, num_fc_layers, basis, theta_size
                        )
                    )
            self.block_list = nn.ModuleList(blocks)
            for i, b in enumerate(blocks):
                setattr(self, f"block{i}", b)

        def forward(self, x):
            residual = x
            forecast = None
            for block in self.block_list:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = (
                    block_forecast if forecast is None else forecast + block_forecast
                )
            return forecast

    torch.manual_seed(0)
    seq_len, pred_len = 24, 8
    stack_types = ("trend", "seasonality")
    num_blocks_per_stack, hidden_dim, num_fc_layers = 2, 16, 2
    trend_degree, num_harmonics = 2, 2

    torch_model = TorchNBeats(
        seq_len,
        pred_len,
        stack_types,
        num_blocks_per_stack,
        hidden_dim,
        num_fc_layers,
        trend_degree,
        num_harmonics,
    )
    torch_model.eval()
    with torch.no_grad():
        for name, p in torch_model.named_parameters():
            p.normal_(0.0, 0.2)

    state_dict = {}
    for k, v in torch_model.state_dict().items():
        if k.startswith("block_list."):
            continue
        state_dict[k] = v.detach().numpy()

    keras_model = NBeats(
        seq_len=seq_len,
        pred_len=pred_len,
        stack_types=stack_types,
        num_blocks_per_stack=num_blocks_per_stack,
        hidden_dim=hidden_dim,
        num_fc_layers=num_fc_layers,
        trend_degree=trend_degree,
        num_harmonics=num_harmonics,
    )

    num_blocks = len(stack_types) * num_blocks_per_stack
    mapper = build_nbeats_mapper(num_blocks=num_blocks, num_fc_layers=num_fc_layers)
    report = WeightConverter(
        keras_model,
        state_dict,
        mapper,
        skip_patterns=[r"\.basis\."],
    ).convert(strict=False, verbose=False)
    assert all("basis" in k for k in report["missing_in_source"])
    assert not report["unused_source_keys"]

    x_np = np.random.randn(3, seq_len).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model(x_np, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert (
        max_diff < 1e-2
    ), f"N-BEATS weight port numerical mismatch: max abs diff {max_diff}"
