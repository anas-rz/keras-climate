import keras
from keras import layers, ops
from keras_climate.forecasting.patchtst import RevIN


class InceptionBlockV1(layers.Layer):

    def __init__(self, out_channels, num_kernels=6, **kwargs):
        super().__init__(**kwargs)
        self.convs = [
            layers.Conv2D(out_channels, 2 * i + 1, padding="same", name=f"kernel{i}")
            for i in range(num_kernels)
        ]

    def call(self, x):
        outs = [conv(x) for conv in self.convs]
        return ops.mean(ops.stack(outs, axis=-1), axis=-1)


class FFTPeriodBlock(layers.Layer):

    def __init__(self, d_model, d_ff, num_kernels=6, top_k=5, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.conv1 = InceptionBlockV1(d_ff, num_kernels, name="conv1")
        self.act = layers.Activation("gelu")
        self.conv2 = InceptionBlockV1(d_model, num_kernels, name="conv2")

    def call(self, x, seq_len):
        B, L, D = ops.shape(x)[0], seq_len, x.shape[-1]

        candidate_periods = self._candidate_periods(seq_len, self.top_k)

        x_time_last = ops.transpose(x, (0, 2, 1))
        real, imag = ops.rfft(x_time_last)
        amplitude = ops.mean(ops.sqrt(ops.square(real) + ops.square(imag)), axis=1)

        results = []
        weights = []
        for period in candidate_periods:
            pad_len = ((seq_len + period - 1) // period) * period
            if pad_len != seq_len:
                pad = ops.zeros((B, pad_len - seq_len, D))
                x_pad = ops.concatenate([x, pad], axis=1)
            else:
                x_pad = x
            num_periods = pad_len // period
            reshaped = ops.reshape(x_pad, (B, num_periods, period, D))

            out = self.conv1(reshaped)
            out = self.act(out)
            out = self.conv2(out)

            out = ops.reshape(out, (B, pad_len, D))[:, :seq_len, :]
            results.append(out)

            freq_idx = min(seq_len // period, seq_len // 2)
            weights.append(amplitude[:, freq_idx])

        stacked = ops.stack(results, axis=-1)
        w = ops.stack(weights, axis=-1)
        w = ops.softmax(w, axis=-1)
        w = w[:, None, None, :]
        fused = ops.sum(stacked * w, axis=-1)
        return fused + x

    @staticmethod
    def _candidate_periods(seq_len, k=5):
        base = [24, 12, 8, 6, 4]
        return [p for p in base[:k] if p < seq_len] or [2]


def TimesNet(
    seq_len=96,
    pred_len=96,
    num_channels=7,
    d_model=64,
    d_ff=128,
    num_layers=2,
    num_kernels=6,
    top_k=5,
    use_revin=True,
    name="timesnet",
):
    inputs = keras.Input(shape=(seq_len, num_channels), name="series")
    x = inputs

    revin = RevIN(name="revin") if use_revin else None
    if revin is not None:
        x = revin(x, mode="norm")

    x = layers.Dense(d_model, name="value_embed")(x)

    for i in range(num_layers):
        x = FFTPeriodBlock(d_model, d_ff, num_kernels, top_k, name=f"timesblock{i}")(x, seq_len=seq_len)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"norm{i}")(x)

    x_t = layers.Permute((2, 1), name="transpose_for_time_proj")(x)
    x_t = layers.Dense(seq_len + pred_len, name="predict_linear")(x_t)
    x = layers.Permute((2, 1), name="transpose_back")(x_t)

    out = layers.Dense(num_channels, name="output_proj")(x)
    out = layers.Lambda(lambda t: t[:, -pred_len:, :], name="slice_horizon")(out)

    if revin is not None:
        out = revin(out, mode="denorm")

    return keras.Model(inputs, out, name=name)
