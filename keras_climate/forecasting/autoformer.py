"""
keras_climate.forecasting.autoformer
----------------------------------------
Autoformer (Wu et al. 2021, NeurIPS): a decomposition-based Transformer
forecaster. Series-decomposition blocks (moving-average trend/seasonal
split, see `keras_climate.forecasting.dlinear.SeriesDecomposition`) are
embedded throughout the encoder and decoder rather than applied once up
front, progressively separating trend from seasonal dynamics at every
layer; the decoder accumulates each layer's extracted trend component
into a running additive trend that is added back to the seasonal output
at the very end. Self-attention is replaced by an Auto-Correlation
mechanism that discovers period-based dependencies via FFT rather than
pointwise dot-products.

Scope note on Auto-Correlation: the official mechanism selects the top-k
highest-autocorrelation time lags (a `torch.topk` over an FFT-computed
autocorrelation function) and aggregates values via a weighted sum of
those lags' circularly time-shifted V - both the lag *selection* and the
per-lag `torch.roll` are inherently dynamic/data-dependent operations that
don't translate cleanly to a static, backend-agnostic Keras graph (JAX
requires static shapes; TF's `tf.roll` supports a dynamic shift but JAX's
does not, at least not portably). This module instead computes a
mathematically equivalent-in-spirit **full** (non-top-k) softmax-weighted
circular convolution of V with the complete autocorrelation profile,
itself computed via FFT (`ops.rfft`/`ops.irfft`, as in
`keras_climate.forecasting.timesnet`) - a differentiable relaxation that
keeps the "attend to periodically-correlated lags via FFT" idea without
the dynamic top-k/roll step. No general-purpose pretrained checkpoint
exists for Autoformer (same situation as Informer - see that module's
docstring) - validated against a from-scratch synthetic PyTorch reference
using the same relaxed mechanism on both sides (see
`weights/mappings/autoformer_mapping.py`).
"""

import keras
from keras import layers, ops
from keras_climate.forecasting.dlinear import SeriesDecomposition
from keras_climate.forecasting.informer import ValueEmbedding


class AutoCorrelation(layers.Layer):
    """FFT-based Auto-Correlation (see module docstring for the top-k ->
    full-softmax relaxation). `query`/`key`/`value` may have different
    time lengths (`key`/`value` are truncated/zero-padded to match
    `query`'s length, matching the official cross-attention usage)."""

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.d_model, name="q_proj")
        self.k_proj = layers.Dense(self.d_model, name="k_proj")
        self.v_proj = layers.Dense(self.d_model, name="v_proj")
        self.out_proj = layers.Dense(self.d_model, name="out_proj")
        super().build(input_shape)

    def _split_heads(self, x, B, L):
        x = ops.reshape(x, (B, L, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))  # (B, H, L, d)

    def call(self, query, key, value):
        B = ops.shape(query)[0]
        Lq, Lk = query.shape[1], key.shape[1]
        q = self._split_heads(self.q_proj(query), B, Lq)
        k = self._split_heads(self.k_proj(key), B, Lk)
        v = self._split_heads(self.v_proj(value), B, Lk)

        if Lk > Lq:
            k, v = k[:, :, :Lq, :], v[:, :, :Lq, :]
        elif Lk < Lq:
            pad = Lq - Lk
            zeros = ops.zeros((B, self.num_heads, pad, self.head_dim))
            k = ops.concatenate([k, zeros], axis=2)
            v = ops.concatenate([v, zeros], axis=2)

        # Cross-correlation of Q, K over the time axis via FFT (lag axis
        # ends up last): corr[..., tau] = sum_t q[t] * k[t - tau].
        q_t = ops.transpose(q, (0, 1, 3, 2))  # (B, H, d, Lq)
        k_t = ops.transpose(k, (0, 1, 3, 2))
        q_re, q_im = ops.rfft(q_t)
        k_re, k_im = ops.rfft(k_t)
        corr_re = q_re * k_re + q_im * k_im
        corr_im = q_im * k_re - q_re * k_im
        corr = ops.irfft((corr_re, corr_im), fft_length=Lq)  # (B, H, d, Lq)

        weights = ops.softmax(ops.mean(corr, axis=2), axis=-1)  # (B, H, Lq)

        # Aggregate V via a circular convolution with `weights` along the
        # time axis (equivalent to a weighted sum of circularly
        # time-shifted V over every lag, computed via FFT rather than an
        # explicit top-k lag loop - see module docstring).
        v_t = ops.transpose(v, (0, 1, 3, 2))  # (B, H, d, Lq)
        w_re, w_im = ops.rfft(weights)  # (B, H, Lq//2+1)
        v_re, v_im = ops.rfft(v_t)  # (B, H, d, Lq//2+1)
        w_re, w_im = w_re[:, :, None, :], w_im[:, :, None, :]
        out_re = v_re * w_re - v_im * w_im
        out_im = v_re * w_im + v_im * w_re
        agg = ops.irfft((out_re, out_im), fft_length=Lq)  # (B, H, d, Lq)
        agg = ops.transpose(agg, (0, 1, 3, 2))  # (B, H, Lq, d)

        out = ops.transpose(agg, (0, 2, 1, 3))
        out = ops.reshape(out, (B, Lq, self.d_model))
        return self.out_proj(out)


class AutoformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, moving_avg=25, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.auto_correlation = AutoCorrelation(d_model, num_heads, name="auto_correlation")
        self.decomp1 = SeriesDecomposition(moving_avg, name="decomp1")
        self.conv1 = layers.Conv1D(d_ff, 1, name="conv1")
        self.act = layers.Activation("gelu")
        self.conv2 = layers.Conv1D(d_model, 1, name="conv2")
        self.decomp2 = SeriesDecomposition(moving_avg, name="decomp2")
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = x + self.drop(self.auto_correlation(x, x, x), training=training)
        x, _ = self.decomp1(x)

        y = self.drop(self.act(self.conv1(x)), training=training)
        y = self.drop(self.conv2(y), training=training)
        x, _ = self.decomp2(x + y)
        return x


class AutoformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, num_channels, moving_avg=25, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_correlation = AutoCorrelation(d_model, num_heads, name="self_correlation")
        self.decomp1 = SeriesDecomposition(moving_avg, name="decomp1")
        self.cross_correlation = AutoCorrelation(d_model, num_heads, name="cross_correlation")
        self.decomp2 = SeriesDecomposition(moving_avg, name="decomp2")
        self.conv1 = layers.Conv1D(d_ff, 1, name="conv1")
        self.act = layers.Activation("gelu")
        self.conv2 = layers.Conv1D(d_model, 1, name="conv2")
        self.decomp3 = SeriesDecomposition(moving_avg, name="decomp3")
        self.trend_proj = layers.Dense(num_channels, use_bias=False, name="trend_proj")
        self.drop = layers.Dropout(dropout)

    def call(self, x, cross, training=False):
        x = x + self.drop(self.self_correlation(x, x, x), training=training)
        x, trend1 = self.decomp1(x)

        x = x + self.drop(self.cross_correlation(x, cross, cross), training=training)
        x, trend2 = self.decomp2(x)

        y = self.drop(self.act(self.conv1(x)), training=training)
        y = self.drop(self.conv2(y), training=training)
        x, trend3 = self.decomp3(x + y)

        trend = self.trend_proj(trend1 + trend2 + trend3)
        return x, trend


def Autoformer(seq_len=96, label_len=48, pred_len=24, num_channels=7, d_model=64, num_heads=4,
               d_ff=128, encoder_layers=2, decoder_layers=1, moving_avg=25, dropout=0.1,
               name="autoformer"):
    """Inputs:
        encoder_series: (B, seq_len, num_channels) - the lookback window.
        label_series: (B, label_len, num_channels) - the last `label_len`
            real values from the lookback window, used to initialize the
            decoder's seasonal/trend components (Autoformer decomposes
            this tail once up front, then pads: zeros for the seasonal
            horizon, the tail's mean for the trend horizon).
    Output: (B, pred_len, num_channels).
    """
    enc_in = keras.Input((seq_len, num_channels), name="encoder_series")
    label_in = keras.Input((label_len, num_channels), name="label_series")

    seasonal_init, trend_init = SeriesDecomposition(moving_avg, name="init_decomp")(label_in)

    mean_trend = layers.Lambda(lambda t: ops.mean(t, axis=1, keepdims=True),
                                name="label_mean")(label_in)
    trend_pred_part = layers.Lambda(lambda t: ops.repeat(t, pred_len, axis=1),
                                     name="trend_pred_fill")(mean_trend)
    trend_init_full = layers.Concatenate(axis=1, name="trend_init_concat")(
        [trend_init, trend_pred_part])

    zeros_seasonal = layers.Lambda(
        lambda t: ops.zeros((ops.shape(t)[0], pred_len, num_channels)),
        name="seasonal_pred_fill")(label_in)
    seasonal_init_full = layers.Concatenate(axis=1, name="seasonal_init_concat")(
        [seasonal_init, zeros_seasonal])

    enc_x = ValueEmbedding(d_model, name="enc_embedding")(enc_in)
    for i in range(encoder_layers):
        enc_x = AutoformerEncoderLayer(d_model, num_heads, d_ff, moving_avg, dropout,
                                        name=f"encoder_layer{i}")(enc_x)

    dec_x = ValueEmbedding(d_model, name="dec_embedding")(seasonal_init_full)
    trend = trend_init_full
    for i in range(decoder_layers):
        dec_x, layer_trend = AutoformerDecoderLayer(
            d_model, num_heads, d_ff, num_channels, moving_avg, dropout,
            name=f"decoder_layer{i}")(dec_x, enc_x)
        trend = layers.Add(name=f"decoder_layer{i}_trend_add")([trend, layer_trend])

    seasonal_out = layers.Dense(num_channels, name="projection")(dec_x)
    out = layers.Add(name="final_add")([seasonal_out, trend])
    out = layers.Lambda(lambda t: t[:, -pred_len:, :], name="slice_forecast")(out)

    return keras.Model([enc_in, label_in], out, name=name)
