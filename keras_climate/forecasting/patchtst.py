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
from keras_climate.utils.layers import TransformerEncoderBlock


class RevIN(layers.Layer):
    """Reversible Instance Normalization (Kim et al. 2022): normalizes each
    series instance over the time axis, and can denormalize predictions.

    `normalize`/`denormalize` are invoked as `revin(x, mode="norm")` /
    `revin(x, mode="denorm")` - going through the real `call()` /
    `__call__()` path, not bypassing it via a couple of custom methods.
    That distinction matters: a layer whose weights are only ever touched
    outside of `__call__` never becomes a node in the Functional graph, so
    Keras has no way to know it belongs to the model - `gamma`/`beta`
    would then silently be missing from `model.weights` entirely, meaning
    `model.fit()` would never update them and `save_weights()` would
    silently drop them."""

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

    def call(self, x, mode):
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got {mode!r}")

    def _normalize(self, x):
        self.mean = ops.mean(x, axis=1, keepdims=True)
        self.std = ops.sqrt(ops.var(x, axis=1, keepdims=True) + self.eps)
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def _denormalize(self, x):
        if self.affine:
            # Matches the official RevIN reference (ts-kim/RevIN, as used
            # by PatchTST) exactly: divides by `gamma + eps**2`, not
            # `gamma + eps` - a deliberate (if minor) asymmetry from the
            # normalize step's `+ eps` under the sqrt.
            x = (x - self.beta) / (self.gamma + self.eps ** 2)
        return x * self.std + self.mean


class LearnedPositionalEmbedding(layers.Layer):
    """A raw (num_patches, d_model) learned positional embedding, added to
    every token.

    Deliberately *not* a `layers.Embedding` lookup over a constant index
    array (`ops.arange(num_patches)`): that array is an eagerly-computed
    constant rather than a symbolic Keras tensor, so calling `Embedding`
    on it never registers as a Functional-graph node - the layer doesn't
    even appear in `model.layers`, and its weights are invisible to
    `model.weights`/`model.fit()`/`save_weights()`. Adding the raw weight
    directly inside `call()`, on the real (symbolic) token tensor, avoids
    that entirely."""

    def __init__(self, num_patches, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.d_model = d_model

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.num_patches, self.d_model), initializer="zeros", name="embeddings")
        super().build(input_shape)

    def call(self, x):
        return x + self.embeddings[None]


class PatchifyTS(layers.Layer):
    """Splits a (B, L, C) series into channel-independent patches:
    output (B*C, num_patches, patch_len).

    `num_patches` is a required (not inferred) argument: computing it from
    a dynamic `ops.shape(x)[1]` and returning it alongside `patches` would
    make this layer's output a mixed tensor/int tuple, which the
    Functional API this layer is used under rejects (a layer's output must
    be tensors only) - it's a static function of `seq_len`/`patch_len`/
    `stride`, all known at model-build time, so the caller passes it in
    directly instead."""

    def __init__(self, patch_len, stride, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches

    def call(self, x):
        # x: (B, L, C) -> (B, C, L) -> patches (B*C, num_patches, patch_len)
        B, C = ops.shape(x)[0], x.shape[-1]
        x = ops.transpose(x, (0, 2, 1))
        idx = (ops.arange(self.patch_len)[None, :]
               + self.stride * ops.arange(self.num_patches)[:, None])
        patches = ops.take(x, idx, axis=2)  # (B, C, num_patches, patch_len)
        patches = ops.reshape(patches, (B * C, self.num_patches, self.patch_len))
        return patches


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
    padding_patch="end",
    name="patchtst",
):
    """`padding_patch="end"` (the official implementation's default, used
    in its published experiments) replicates the last timestep `stride`
    times before patching, so the final `stride`-sized tail of the series
    - which would otherwise be dropped by non-overlapping-stride patching
    - still contributes one more patch. Pass `padding_patch=None` for the
    simpler no-padding variant (one fewer patch)."""
    inputs = keras.Input(shape=(seq_len, num_channels), name="series")
    x = inputs

    revin = RevIN(name="revin") if use_revin else None
    if revin is not None:
        x = revin(x, mode="norm")

    padded_seq_len = seq_len
    if padding_patch == "end":
        padded_seq_len += stride
        x = layers.Lambda(
            lambda t: ops.concatenate([t, ops.repeat(t[:, -1:, :], stride, axis=1)], axis=1),
            name="replication_pad",
        )(x)

    num_patches = (padded_seq_len - patch_len) // stride + 1
    patches = PatchifyTS(patch_len, stride, num_patches, name="patchify")(x)

    tokens = layers.Dense(d_model, name="patch_proj")(patches)
    tokens = LearnedPositionalEmbedding(num_patches, d_model, name="pos_embed")(tokens)
    tokens = layers.Dropout(dropout)(tokens)

    for i in range(depth):
        tokens = TransformerEncoderBlock(d_model, num_heads, mlp_ratio, drop=dropout,
                                          name=f"encoder_block{i}")(tokens)
    tokens = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(tokens)

    flat = layers.Reshape((num_patches * d_model,), name="flatten")(tokens)
    head = layers.Dense(pred_len, name="forecast_head")(flat)  # (B*C, pred_len)

    out = layers.Lambda(
        lambda t: ops.transpose(ops.reshape(t, (-1, num_channels, pred_len)), (0, 2, 1)),
        name="reshape_output",
    )(head)

    if revin is not None:
        out = revin(out, mode="denorm")

    return keras.Model(inputs, out, name=name)
