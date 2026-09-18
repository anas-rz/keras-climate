"""
keras_climate.forecasting.dlinear
-------------------------------------
DLinear (Zeng et al. 2022, "Are Transformers Effective for Time Series
Forecasting?"): decomposes the input into a trend (moving-average) and a
seasonal (residual) component, each passed through its own single Linear
layer over the time axis, then summed - a deliberately minimal baseline
that outperformed several contemporary Transformer forecasters on common
long-horizon benchmarks. No public general-purpose pretrained checkpoint
exists (the official repo only ships per-benchmark-dataset training
scripts/configs, not a reusable backbone checkpoint) - validated against
a from-scratch synthetic PyTorch reference matching the official repo's
own module naming (`Linear_Seasonal`/`Linear_Trend`).
"""

import keras
from keras import layers, ops


class SeriesDecomposition(layers.Layer):
    """Splits `(B, L, C)` into `(seasonal, trend)` via a moving-average
    trend estimate. Edge-replicated padding before pooling matches the
    official implementation's `nn.AvgPool1d` + replicate-pad."""

    def __init__(self, kernel_size=25, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.avg_pool = layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding="valid")

    def call(self, x):
        pad_front = (self.kernel_size - 1) // 2
        pad_end = self.kernel_size - 1 - pad_front
        front = ops.repeat(x[:, :1, :], pad_front, axis=1)
        end = ops.repeat(x[:, -1:, :], pad_end, axis=1)
        padded = ops.concatenate([front, x, end], axis=1)
        trend = self.avg_pool(padded)
        seasonal = x - trend
        return seasonal, trend


def _time_linear(x, pred_len, num_channels, individual, name):
    """Applies a Linear (seq_len -> pred_len) over the time axis - either
    one shared Dense across all channels, or one Dense per channel
    (`individual=True`, matching the official `nn.ModuleList` variant)."""
    xt = layers.Permute((2, 1))(x)  # (B, C, L)
    if individual:
        per_channel = [
            layers.Dense(pred_len, name=f"{name}{c}")(
                layers.Lambda(lambda t, idx=c: t[:, idx:idx + 1, :])(xt))
            for c in range(num_channels)
        ]
        out = layers.Concatenate(axis=1)(per_channel) if num_channels > 1 else per_channel[0]
    else:
        out = layers.Dense(pred_len, name=name)(xt)
    return layers.Permute((2, 1))(out)  # (B, pred_len, C)


def DLinear(seq_len=336, pred_len=96, num_channels=7, kernel_size=25,
            individual=False, name="dlinear"):
    inputs = keras.Input(shape=(seq_len, num_channels), name="series")
    seasonal, trend = SeriesDecomposition(kernel_size, name="decomp")(inputs)

    seasonal_out = _time_linear(seasonal, pred_len, num_channels, individual, "linear_seasonal")
    trend_out = _time_linear(trend, pred_len, num_channels, individual, "linear_trend")
    outputs = layers.Add(name="add")([seasonal_out, trend_out])

    return keras.Model(inputs, outputs, name=name)
