import keras
from keras import layers, ops


class SeriesDecomposition(layers.Layer):

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
    xt = layers.Permute((2, 1))(x)
    if individual:
        per_channel = [
            layers.Dense(pred_len, name=f"{name}{c}")(
                layers.Lambda(lambda t, idx=c: t[:, idx:idx + 1, :])(xt))
            for c in range(num_channels)
        ]
        out = layers.Concatenate(axis=1)(per_channel) if num_channels > 1 else per_channel[0]
    else:
        out = layers.Dense(pred_len, name=name)(xt)
    return layers.Permute((2, 1))(out)


def DLinear(seq_len=336, pred_len=96, num_channels=7, kernel_size=25,
            individual=False, name="dlinear"):
    inputs = keras.Input(shape=(seq_len, num_channels), name="series")
    seasonal, trend = SeriesDecomposition(kernel_size, name="decomp")(inputs)

    seasonal_out = _time_linear(seasonal, pred_len, num_channels, individual, "linear_seasonal")
    trend_out = _time_linear(trend, pred_len, num_channels, individual, "linear_trend")
    outputs = layers.Add(name="add")([seasonal_out, trend_out])

    return keras.Model(inputs, outputs, name=name)
