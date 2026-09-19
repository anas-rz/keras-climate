import keras
from keras import layers, ops
from keras_climate.utils.layers import TransformerEncoderBlock


class RevIN(layers.Layer):

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
            x = (x - self.beta) / (self.gamma + self.eps**2)
        return x * self.std + self.mean


class LearnedPositionalEmbedding(layers.Layer):

    def __init__(self, num_patches, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.d_model = d_model

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.num_patches, self.d_model),
            initializer="zeros",
            name="embeddings",
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.embeddings[None]


class PatchifyTS(layers.Layer):

    def __init__(self, patch_len, stride, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = num_patches

    def call(self, x):
        B, C = ops.shape(x)[0], x.shape[-1]
        x = ops.transpose(x, (0, 2, 1))
        idx = (
            ops.arange(self.patch_len)[None, :]
            + self.stride * ops.arange(self.num_patches)[:, None]
        )
        patches = ops.take(x, idx, axis=2)
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
    inputs = keras.Input(shape=(seq_len, num_channels), name="series")
    x = inputs

    revin = RevIN(name="revin") if use_revin else None
    if revin is not None:
        x = revin(x, mode="norm")

    padded_seq_len = seq_len
    if padding_patch == "end":
        padded_seq_len += stride
        x = layers.Lambda(
            lambda t: ops.concatenate(
                [t, ops.repeat(t[:, -1:, :], stride, axis=1)], axis=1
            ),
            name="replication_pad",
        )(x)

    num_patches = (padded_seq_len - patch_len) // stride + 1
    patches = PatchifyTS(patch_len, stride, num_patches, name="patchify")(x)

    tokens = layers.Dense(d_model, name="patch_proj")(patches)
    tokens = LearnedPositionalEmbedding(num_patches, d_model, name="pos_embed")(tokens)
    tokens = layers.Dropout(dropout)(tokens)

    for i in range(depth):
        tokens = TransformerEncoderBlock(
            d_model, num_heads, mlp_ratio, drop=dropout, name=f"encoder_block{i}"
        )(tokens)
    tokens = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(tokens)

    flat = layers.Reshape((num_patches * d_model,), name="flatten")(tokens)
    head = layers.Dense(pred_len, name="forecast_head")(flat)

    out = layers.Lambda(
        lambda t: ops.transpose(
            ops.reshape(t, (-1, num_channels, pred_len)), (0, 2, 1)
        ),
        name="reshape_output",
    )(head)

    if revin is not None:
        out = revin(out, mode="denorm")

    return keras.Model(inputs, out, name=name)
