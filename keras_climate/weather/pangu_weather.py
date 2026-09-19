import numpy as np
import keras
from keras import layers, ops

WINDOW_SIZE = (2, 6, 12)


def _pad_to_multiple(n, m):
    return (m - n % m) % m


class PatchEmbedding(layers.Layer):

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.conv = layers.Dense(self.dim, name="conv")
        self.conv_surface = layers.Dense(self.dim, name="conv_surface")
        super().build(input_shape)

    def call(self, inputs):
        upper, surface = inputs
        B = ops.shape(upper)[0]

        upper = ops.pad(upper, [[0, 0], [0, 1], [0, 3], [0, 0], [0, 0]])
        upper = ops.reshape(upper, (B, 7, 2, 181, 4, 360, 4, 6))
        upper = ops.transpose(upper, (0, 1, 3, 5, 7, 2, 4, 6))
        upper = ops.reshape(upper, (B, 7, 181, 360, 6 * 2 * 4 * 4))
        upper = self.conv(upper)

        surface = ops.pad(surface, [[0, 0], [0, 3], [0, 0], [0, 0]])
        surface = ops.reshape(surface, (B, 181, 4, 360, 4, 7))
        surface = ops.transpose(surface, (0, 1, 3, 5, 2, 4))
        surface = ops.reshape(surface, (B, 181, 360, 7 * 4 * 4))
        surface = self.conv_surface(surface)
        surface = surface[:, None]

        x = ops.concatenate([surface, upper], axis=1)
        return ops.reshape(x, (B, 8 * 181 * 360, self.dim))


class EarthMlp(layers.Layer):

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.linear1 = layers.Dense(dim * 4, activation="gelu", name="linear1")
        self.linear2 = layers.Dense(dim, name="linear2")

    def call(self, x):
        return self.linear2(self.linear1(x))


class EarthAttention3D(layers.Layer):

    def __init__(self, dim, heads, type_of_windows, window_size=WINDOW_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.type_of_windows = type_of_windows
        self.window_vol = window_size[0] * window_size[1] * window_size[2]

    def build(self, input_shape):
        self.linear1 = layers.Dense(self.dim * 3, name="linear1")
        self.linear2 = layers.Dense(self.dim, name="linear2")
        self.earth_specific_bias = self.add_weight(
            shape=(1, self.type_of_windows, self.heads, self.window_vol, self.window_vol),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True, name="earth_specific_bias")
        super().build(input_shape)

    def call(self, x, attn_mask=None):
        # x: (nWB, type_of_windows, window_vol, C), nWB = B * num_longitude_windows
        #
        # NOTE: this parameter is deliberately named `attn_mask`, not
        # `mask` - Keras's `Layer.__call__` reserves the literal name
        # `mask` for its own automatic mask-propagation protocol (see the
        # harmless "was passed an input with a mask attached" warning
        # this layer would otherwise trigger). Using the reserved name
        # wasn't actually the source of a real bug found during
        # development here (a since-fixed test-data-construction issue
        # was), but it's a genuine footgun worth avoiding on principle -
        # a custom `call()` kwarg named `mask` can be silently intercepted
        # by the framework instead of reaching your own code as a plain
        # value.
        nWB = ops.shape(x)[0]
        T, N, C = self.type_of_windows, self.window_vol, self.dim
        qkv = self.linear1(x)
        qkv = ops.reshape(qkv, (nWB, T, N, 3, self.heads, self.head_dim))
        qkv = ops.transpose(qkv, (3, 0, 1, 4, 2, 5))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 2, 4, 3)))
        attn = attn + self.earth_specific_bias

        if attn_mask is not None:
            nWp = ops.shape(attn_mask)[0]
            attn = ops.reshape(attn, (nWB // nWp, nWp, T, self.heads, N, N))
            attn = attn + attn_mask[None, :, :, None, :, :]
            attn = ops.reshape(attn, (nWB, T, self.heads, N, N))

        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 1, 3, 2, 4))
        out = ops.reshape(out, (nWB, T, N, C))
        return self.linear2(out)


def _gen_shift_mask(Z, H_pad, W, window_size):
    wZ, wH, wW = window_size
    img_mask = np.zeros((1, Z, H_pad, W, 1), dtype="float32")
    z_slices = (slice(0, -wZ), slice(-wZ, -wZ // 2), slice(-wZ // 2, None))
    h_slices = (slice(0, -wH), slice(wH, -wH // 2), slice(-wH // 2, None))
    cnt = 0
    for z in z_slices:
        for h in h_slices:
            img_mask[:, z, h, :, :] = cnt
            cnt += 1
    img_mask = img_mask.reshape(1, Z // wZ, wZ, H_pad // wH, wH, W // wW, wW, 1)
    img_mask = img_mask.transpose(0, 5, 1, 3, 2, 4, 6, 7)
    type_of_windows = (Z // wZ) * (H_pad // wH)
    mask_windows = img_mask.reshape(-1, type_of_windows, wZ * wH * wW)
    attn_mask = mask_windows[:, :, None, :] - mask_windows[:, :, :, None]
    return np.where(attn_mask != 0, -100.0, 0.0).astype("float32")


class EarthSpecificBlock(layers.Layer):

    def __init__(self, dim, heads, resolution, shift, window_size=WINDOW_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.shift = shift
        Z, H, W = resolution
        self.Z, self.H, self.W = Z, H, W
        wZ, wH, wW = window_size
        self.H_pad = H + _pad_to_multiple(H, wH)
        self.type_of_windows = (Z // wZ) * (self.H_pad // wH)

        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.attention = EarthAttention3D(dim, heads, self.type_of_windows, window_size, name="attention")
        self.linear = EarthMlp(dim, name="linear")

        self._mask_np = _gen_shift_mask(Z, self.H_pad, W, window_size) if shift else None

    def build(self, input_shape):
        if self._mask_np is not None:
            self.attn_mask = self.add_weight(
                shape=self._mask_np.shape, initializer=keras.initializers.Constant(self._mask_np),
                trainable=False, name="attn_mask")
        else:
            self.attn_mask = None
        super().build(input_shape)

    def call(self, x, training=False):
        B = ops.shape(x)[0]
        Z, H, W, C = self.Z, self.H, self.W, self.dim
        wZ, wH, wW = self.window_size
        H_pad = self.H_pad
        shortcut = x

        x = ops.reshape(x, (B, Z, H, W, C))
        x = ops.pad(x, [[0, 0], [0, 0], [0, H_pad - H], [0, 0], [0, 0]])

        if self.shift:
            x = ops.roll(x, shift=(-wZ // 2, -wH // 2, -wW // 2), axis=(1, 2, 3))

        x = ops.reshape(x, (B, Z // wZ, wZ, H_pad // wH, wH, W // wW, wW, C))
        x = ops.transpose(x, (0, 5, 1, 3, 2, 4, 6, 7))
        x = ops.reshape(x, (B * (W // wW), self.type_of_windows, wZ * wH * wW, C))

        x = self.attention(x, attn_mask=self.attn_mask)

        x = ops.reshape(x, (B, W // wW, Z // wZ, H_pad // wH, wZ, wH, wW, C))
        x = ops.transpose(x, (0, 2, 4, 3, 5, 1, 6, 7))
        x = ops.reshape(x, (B, Z, H_pad, W, C))

        if self.shift:
            x = ops.roll(x, shift=(wZ // 2, wH // 2, wW // 2), axis=(1, 2, 3))

        x = x[:, :, :H, :, :]
        x = ops.reshape(x, (B, Z * H * W, C))

        x = shortcut + self.norm1(x)
        x = x + self.norm2(self.linear(x))
        return x


class EarthSpecificLayer(layers.Layer):

    def __init__(self, depth, dim, heads, resolution, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            EarthSpecificBlock(dim, heads, resolution, shift=(i % 2 == 1),
                                name=f"blocks_EarthSpecificBlock{i}")
            for i in range(depth)
        ]

    def call(self, x, training=False):
        for blk in self.blocks:
            x = blk(x, training=training)
        return x


class DownSample(layers.Layer):

    def __init__(self, dim, resolution=(8, 181, 360), **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.Z, self.H, self.W = resolution

    def build(self, input_shape):
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.linear = layers.Dense(2 * self.dim, use_bias=False, name="linear")
        super().build(input_shape)

    def call(self, x):
        B = ops.shape(x)[0]
        Z, H, W, C = self.Z, self.H, self.W, self.dim
        x = ops.reshape(x, (B, Z, H, W, C))
        x = ops.pad(x, [[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])
        H2 = H + 1
        x = ops.reshape(x, (B, Z, H2 // 2, 2, W // 2, 2, C))
        x = ops.transpose(x, (0, 1, 2, 4, 3, 5, 6))
        x = ops.reshape(x, (B, Z * (H2 // 2) * (W // 2), 4 * C))
        return self.linear(self.norm(x))


class UpSample(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.linear1 = layers.Dense(self.output_dim * 4, use_bias=False, name="linear1")
        self.linear2 = layers.Dense(self.output_dim, use_bias=False, name="linear2")
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        super().build(input_shape)

    def call(self, x):
        B = ops.shape(x)[0]
        x = self.linear1(x)
        x = ops.reshape(x, (B, 8, 91, 180, 2, 2, self.output_dim))
        x = ops.transpose(x, (0, 1, 2, 4, 3, 5, 6))
        x = ops.reshape(x, (B, 8, 182, 360, self.output_dim))
        x = x[:, :, :181, :, :]
        x = ops.reshape(x, (B, 8 * 181 * 360, self.output_dim))
        return self.linear2(self.norm(x))


class PatchRecovery(layers.Layer):

    def build(self, input_shape):
        self.conv = layers.Dense(5 * 2 * 4 * 4, name="conv")
        self.conv_surface = layers.Dense(4 * 4 * 4, name="conv_surface")
        super().build(input_shape)

    def call(self, x):
        B = ops.shape(x)[0]
        C = x.shape[-1]
        x = ops.reshape(x, (B, 8, 181, 360, C))

        upper = self.conv(x[:, 1:])
        upper = ops.reshape(upper, (B, 7, 181, 360, 5, 2, 4, 4))
        upper = ops.transpose(upper, (0, 4, 1, 5, 2, 6, 3, 7))
        upper = ops.reshape(upper, (B, 5, 14, 724, 1440))
        upper = upper[:, :, :13, :721, :]

        surface = self.conv_surface(x[:, 0])
        surface = ops.reshape(surface, (B, 181, 360, 4, 4, 4))
        surface = ops.transpose(surface, (0, 3, 1, 4, 2, 5))
        surface = ops.reshape(surface, (B, 4, 724, 1440))
        surface = surface[:, :, :721, :]

        return upper, surface


def PanguWeather(depths=(2, 6, 6, 2), num_heads=(6, 12, 12, 6), dims=(192, 384, 384, 192),
                  name="pangu_weather"):
    upper_in = keras.Input(shape=(13, 721, 1440, 6), name="upper_air")
    surface_in = keras.Input(shape=(721, 1440, 7), name="surface")

    x = PatchEmbedding(dims[0], name="_input_layer")([upper_in, surface_in])

    x = EarthSpecificLayer(depths[0], dims[0], num_heads[0], (8, 181, 360),
                            name="layers_EarthSpecificLayer0")(x)
    skip = x

    x = DownSample(dims[0], (8, 181, 360), name="downsample")(x)

    x = EarthSpecificLayer(depths[1], dims[1], num_heads[1], (8, 91, 180),
                            name="layers_EarthSpecificLayer1")(x)
    x = EarthSpecificLayer(depths[2], dims[2], num_heads[2], (8, 91, 180),
                            name="layers_EarthSpecificLayer2")(x)

    x = UpSample(dims[3], name="upsample")(x)

    x = EarthSpecificLayer(depths[3], dims[3], num_heads[3], (8, 181, 360),
                            name="layers_EarthSpecificLayer3")(x)

    x = layers.Concatenate(axis=-1, name="skip_concat")([skip, x])

    upper_out, surface_out = PatchRecovery(name="_output_layer")(x)

    return keras.Model([upper_in, surface_in], [upper_out, surface_out], name=name)
