"""
keras_climate.forecasting.patchtst
-------------------------------------
PatchTST (Nie et al. 2023): channel-independent patch-based transformer for
long-horizon multivariate time-series forecasting. Each variate (e.g. a
climate station's temperature, humidity, pressure channel) is patched and
encoded independently by a shared transformer, then linearly projected to
the forecast horizon.
"""

import keras
from keras import layers, ops
from ..utils.layers import TransformerEncoderBlock


class RevIN(layers.Layer):
    """Reversible Instance Normalization (Kim et al. 2022): normalizes each
    series instance over the time axis, and can denormalize predictions."""

    def __init__(self, eps=1e-5, affine=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        if self.affine:
            C = input_shape[-1]
            self.gamma = self.add_weight(shape=(C,), initializer="ones", name="gamma")
            self.beta = self.add_weight(shape=(C,), initializer="zeros", name="beta")
        super().build(input_shape)

    def normalize(self, x):
        self.mean = ops.mean(x, axis=1, keepdims=True)
        self.std = ops.sqrt(ops.var(x, axis=1, keepdims=True) + self.eps)
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denormalize(self, x):
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)
        return x * self.std + self.mean


class PatchifyTS(layers.Layer):
    """Splits a (B, L, C) series into channel-independent patches:
    output (B*C, num_patches, patch_len)."""

    def __init__(self, patch_len, stride, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride

    def call(self, x):
        # x: (B, L, C) -> (B, C, L) -> patches (B, C, num_patches, patch_len)
        B, L, C = ops.shape(x)[0], ops.shape(x)[1], x.shape[-1]
        x = ops.transpose(x, (0, 2, 1))
        num_patches = (L - self.patch_len) // self.stride + 1
        idx = ops.arange(self.patch_len)[None, :] + self.stride * ops.arange(num_patches)[:, None]
        patches = ops.take(x, idx, axis=2)  # (B, C, num_patches, patch_len)
        patches = ops.reshape(patches, (B * C, num_patches, self.patch_len))
        return patches, B, C, num_patches


def PatchTST(
    seq_len=336,
    pred_len=96,
    num_channels=7,
    patch_len=16,
    stride=8,
    d_model=128,
    depth=3,
    num_heads=16,
    mlp_ratio=2.0,
    dropout=0.2,
    use_revin=True,
    name="patchtst",
):
    inputs = keras.Input(shape=(seq_len, num_channels), name="series")
    x = inputs

    revin = RevIN(name="revin") if use_revin else None
    if revin is not None:
        x = revin.normalize(x)

    patches, B, C, num_patches = PatchifyTS(patch_len, stride, name="patchify")(x)

    tokens = layers.Dense(d_model, name="patch_proj")(patches)
    pos_embed = layers.Embedding(num_patches, d_model, name="pos_embed")(ops.arange(num_patches))
    tokens = tokens + pos_embed[None]
    tokens = layers.Dropout(dropout)(tokens)

    for i in range(depth):
        tokens = TransformerEncoderBlock(d_model, num_heads, mlp_ratio, drop=dropout,
                                          name=f"encoder_block{i}")(tokens)
    tokens = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(tokens)

    flat = layers.Reshape((num_patches * d_model,), name="flatten")(tokens)
    head = layers.Dense(pred_len, name="forecast_head")(flat)  # (B*C, pred_len)

    out = layers.Lambda(
        lambda t: ops.transpose(ops.reshape(t, (B, C, pred_len)), (0, 2, 1)),
        name="reshape_output",
    )(head)

    if revin is not None:
        out = revin.denormalize(out)

    return keras.Model(inputs, out, name=name)
