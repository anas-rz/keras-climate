"""
keras_climate.forecasting.nbeats
------------------------------------
N-BEATS (Oreshkin et al. 2020, "Neural Basis Expansion Analysis for
Interpretable Time Series Forecasting"): a deep stack of fully-connected
blocks. Each block maps the lookback window to a small set of "theta"
coefficients that parameterize an interpretable basis expansion (generic,
polynomial trend, or Fourier seasonality), producing both a "backcast"
(reconstruction of its input, subtracted from the residual seen by later
blocks) and a "forecast" (added into the running forecast total) - the
paper's "doubly residual stacking". Operates on a single (univariate)
series per call, as in the original paper; apply per-channel for
multivariate series.

No general-purpose pretrained checkpoint exists for N-BEATS (the official
repo and community reimplementations only ship per-benchmark-dataset
training scripts) - validated against a from-scratch synthetic PyTorch
reference of this repo's own basis parameterization (see
`weights/mappings/nbeats_mapping.py`).
"""

import numpy as np
import keras
from keras import layers, ops


class GenericBasis(layers.Layer):
    """theta split directly into (backcast, forecast) - no fixed basis
    function, fully learned via the two theta chunks."""

    def __init__(self, backcast_size, forecast_size, **kwargs):
        super().__init__(**kwargs)
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def call(self, theta):
        return theta[:, :self.backcast_size], theta[:, self.backcast_size:]


class TrendBasis(layers.Layer):
    """Low-degree polynomial basis over the normalized time index.
    `theta_size = 2 * (degree + 1)`: the first half are forecast
    coefficients, the second half backcast coefficients."""

    def __init__(self, degree, backcast_size, forecast_size, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        poly = degree + 1
        self._backcast_basis_np = np.stack(
            [(np.arange(backcast_size) / backcast_size) ** i for i in range(poly)]).astype("float32")
        self._forecast_basis_np = np.stack(
            [(np.arange(forecast_size) / forecast_size) ** i for i in range(poly)]).astype("float32")

    def build(self, input_shape):
        self.backcast_basis = self.add_weight(
            shape=self._backcast_basis_np.shape,
            initializer=keras.initializers.Constant(self._backcast_basis_np),
            trainable=False, name="backcast_basis")
        self.forecast_basis = self.add_weight(
            shape=self._forecast_basis_np.shape,
            initializer=keras.initializers.Constant(self._forecast_basis_np),
            trainable=False, name="forecast_basis")
        super().build(input_shape)

    def call(self, theta):
        poly = self.degree + 1
        forecast_theta, backcast_theta = theta[:, :poly], theta[:, poly:]
        backcast = ops.matmul(backcast_theta, self.backcast_basis)
        forecast = ops.matmul(forecast_theta, self.forecast_basis)
        return backcast, forecast


class SeasonalityBasis(layers.Layer):
    """Fourier basis with harmonic frequencies scaled by the forecast
    horizon. `theta_size = 4 * num_harmonics`: forecast-cos, forecast-sin,
    backcast-cos, backcast-sin, each `num_harmonics` coefficients."""

    def __init__(self, num_harmonics, backcast_size, forecast_size, **kwargs):
        super().__init__(**kwargs)
        self.num_harmonics = num_harmonics
        k = np.arange(1, num_harmonics + 1)[:, None]
        t_back = np.arange(backcast_size)[None, :] / forecast_size
        t_fore = np.arange(forecast_size)[None, :] / forecast_size
        self._backcast_cos_np = np.cos(2 * np.pi * k * t_back).astype("float32")
        self._backcast_sin_np = np.sin(2 * np.pi * k * t_back).astype("float32")
        self._forecast_cos_np = np.cos(2 * np.pi * k * t_fore).astype("float32")
        self._forecast_sin_np = np.sin(2 * np.pi * k * t_fore).astype("float32")

    def build(self, input_shape):
        def const(arr, name):
            return self.add_weight(shape=arr.shape, initializer=keras.initializers.Constant(arr),
                                    trainable=False, name=name)

        self.backcast_cos = const(self._backcast_cos_np, "backcast_cos")
        self.backcast_sin = const(self._backcast_sin_np, "backcast_sin")
        self.forecast_cos = const(self._forecast_cos_np, "forecast_cos")
        self.forecast_sin = const(self._forecast_sin_np, "forecast_sin")
        super().build(input_shape)

    def call(self, theta):
        H = self.num_harmonics
        fc, fs = theta[:, :H], theta[:, H:2 * H]
        bc, bs = theta[:, 2 * H:3 * H], theta[:, 3 * H:4 * H]
        forecast = ops.matmul(fc, self.forecast_cos) + ops.matmul(fs, self.forecast_sin)
        backcast = ops.matmul(bc, self.backcast_cos) + ops.matmul(bs, self.backcast_sin)
        return backcast, forecast


class NBeatsBlock(layers.Layer):
    """`num_fc_layers` ReLU-Dense layers -> theta -> basis(theta)."""

    def __init__(self, hidden_dim, num_fc_layers, basis, theta_size, **kwargs):
        super().__init__(**kwargs)
        self.fc_layers = [layers.Dense(hidden_dim, activation="relu", name=f"fc{i}")
                           for i in range(num_fc_layers)]
        self.theta_layer = layers.Dense(theta_size, name="theta")
        self.basis = basis

    def call(self, x):
        h = x
        for fc in self.fc_layers:
            h = fc(h)
        theta = self.theta_layer(h)
        return self.basis(theta)


def _make_basis(stack_type, seq_len, pred_len, trend_degree, num_harmonics, name):
    if stack_type == "generic":
        return GenericBasis(seq_len, pred_len, name=name), seq_len + pred_len
    if stack_type == "trend":
        return TrendBasis(trend_degree, seq_len, pred_len, name=name), 2 * (trend_degree + 1)
    if stack_type == "seasonality":
        return SeasonalityBasis(num_harmonics, seq_len, pred_len, name=name), 4 * num_harmonics
    raise ValueError(f"Unknown stack_type '{stack_type}'")


def NBeats(seq_len=96, pred_len=24, stack_types=("trend", "seasonality"), num_blocks_per_stack=3,
           hidden_dim=256, num_fc_layers=4, trend_degree=2, num_harmonics=1, name="nbeats"):
    """Univariate forecaster: `(B, seq_len) -> (B, pred_len)`."""
    inputs = keras.Input(shape=(seq_len,), name="series")
    residual = inputs
    forecast = None

    block_idx = 0
    for stack_type in stack_types:
        for _ in range(num_blocks_per_stack):
            basis, theta_size = _make_basis(stack_type, seq_len, pred_len, trend_degree, num_harmonics,
                                             name=f"block{block_idx}_basis")
            block = NBeatsBlock(hidden_dim, num_fc_layers, basis, theta_size, name=f"block{block_idx}")
            backcast, block_forecast = block(residual)
            residual = layers.Subtract(name=f"block{block_idx}_residual")([residual, backcast])
            forecast = (block_forecast if forecast is None
                        else layers.Add(name=f"block{block_idx}_forecast_sum")([forecast, block_forecast]))
            block_idx += 1

    return keras.Model(inputs, forecast, name=name)
