import math

import numpy as np
import keras
from keras import layers, ops

_LN_EPS = 1e-5


def posemb_sincos_1d(num_or_values, dim, temperature=10000.0):
    values = (
        np.arange(num_or_values, dtype=np.float32)
        if isinstance(num_or_values, int)
        else np.asarray(num_or_values, dtype=np.float32)
    )
    omega = np.arange(dim // 2, dtype=np.float32) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)
    scaled = values[:, None] * omega[None, :]
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)


def posemb_sincos_2d_with_gsd(h, w, dim, gsd=1.0, temperature=10000.0):
    assert dim % 4 == 0
    y, x = np.meshgrid(
        np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
    )
    omega = np.arange(dim // 4, dtype=np.float32) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * gsd
    y = y.reshape(-1)[:, None] * omega[None, :]
    x = x.reshape(-1)[:, None] * omega[None, :]
    return np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1).astype(
        np.float32
    )


class FCBlock(layers.Layer):

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def build(self, input_shape):
        self.l1 = layers.Dense(self.size, name="l1")
        self.l2 = layers.Dense(self.size, name="l2")
        super().build(input_shape)

    def call(self, x):
        y = keras.activations.gelu(self.l1(x))
        y = keras.activations.gelu(self.l2(y))
        return x + y


class _TorchStyleEncoderLayer(layers.Layer):

    def __init__(self, dim, num_heads, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.in_proj = layers.Dense(self.dim * 3, name="in_proj")
        self.out_proj = layers.Dense(self.dim, name="out_proj")
        self.norm1 = layers.LayerNormalization(epsilon=_LN_EPS, name="norm1")
        self.linear1 = layers.Dense(self.mlp_dim, name="linear1")
        self.linear2 = layers.Dense(self.dim, name="linear2")
        self.norm2 = layers.LayerNormalization(epsilon=_LN_EPS, name="norm2")
        super().build(input_shape)

    def call(self, x):
        B, N = ops.shape(x)[0], ops.shape(x)[1]
        qkv = self.in_proj(x)
        qkv = ops.transpose(
            ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.softmax(
            ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale, axis=-1
        )
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, N, self.dim))
        out = self.out_proj(out)
        x = self.norm1(x + out)
        ff = self.linear2(keras.activations.gelu(self.linear1(x)))
        x = self.norm2(x + ff)
        return x


class WavesTransformer(layers.Layer):

    def __init__(
        self, wave_dim, output_dim, num_latent_tokens, embed_dim, num_heads=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.wave_dim = wave_dim
        self.output_dim = output_dim
        self.num_latent_tokens = num_latent_tokens
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def build(self, input_shape):
        self.encoder_layer = _TorchStyleEncoderLayer(
            self.wave_dim, self.num_heads, 2048, name="encoder_layer"
        )
        self.fc_weight = layers.Dense(self.output_dim, name="fc_weight")
        self.fc_bias = layers.Dense(self.embed_dim, name="fc_bias")
        self.weight_tokens = self.add_weight(
            shape=(self.num_latent_tokens, self.wave_dim),
            initializer="random_normal",
            name="weight_tokens",
        )
        self.bias_token = self.add_weight(
            shape=(1, self.wave_dim), initializer="random_normal", name="bias_token"
        )
        super().build(input_shape)

    def call(self, x):
        seq = ops.concatenate([self.weight_tokens, x, self.bias_token], axis=0)[None]
        out = self.encoder_layer(seq)[0]
        num_bands = ops.shape(x)[0]
        wave_slice = (
            out[self.num_latent_tokens : self.num_latent_tokens + num_bands] + x
        )
        weights = self.fc_weight(wave_slice)
        bias = self.fc_bias(out[-1:])[0]
        return weights, bias


class DynamicEmbedding(layers.Layer):

    def __init__(self, wave_dim, num_latent_tokens, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = patch_size * patch_size * embed_dim

    def build(self, input_shape):
        self.weight_generator = WavesTransformer(
            self.wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
            name="weight_generator",
        )
        self.fclayer = FCBlock(self.wave_dim, name="fclayer")
        super().build(input_shape)

    def call(self, cube, waves):
        waves_enc = _sincos_1d(waves, self.wave_dim)
        waves_enc = self.fclayer(waves_enc)
        weight, bias = self.weight_generator(waves_enc)

        p = self.patch_size
        num_bands = ops.shape(cube)[-1]
        kernel = ops.reshape(weight, (num_bands, self.embed_dim, p, p))
        kernel = ops.transpose(kernel, (2, 3, 0, 1))

        out = ops.conv(cube, kernel * 0.02, strides=(p, p), padding="valid")
        out = out + bias * 0.02
        B = ops.shape(out)[0]
        H, W = ops.shape(out)[1], ops.shape(out)[2]
        return ops.reshape(out, (B, H * W, self.embed_dim)), waves_enc


def _sincos_1d(waves, dim, temperature=10000.0):
    omega = ops.arange(dim // 2, dtype="float32") / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)
    scaled = waves[:, None] * omega[None, :]
    return ops.concatenate([ops.sin(scaled), ops.cos(scaled)], axis=1)


class ClayTransformerBlock(layers.Layer):

    def __init__(self, dim, num_heads, dim_head, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * num_heads
        self.scale = dim_head**-0.5
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.attn_norm = layers.LayerNormalization(epsilon=_LN_EPS, name="attn_norm")
        self.to_qkv = layers.Dense(self.inner_dim * 3, use_bias=False, name="to_qkv")
        self.to_out = layers.Dense(self.dim, use_bias=False, name="to_out")
        self.ff_norm = layers.LayerNormalization(epsilon=_LN_EPS, name="ff_norm")
        self.ff1 = layers.Dense(self.mlp_dim, name="ff1")
        self.ff2 = layers.Dense(self.dim, name="ff2")
        super().build(input_shape)

    def call(self, x):
        B, N = ops.shape(x)[0], ops.shape(x)[1]
        xn = self.attn_norm(x)
        qkv = self.to_qkv(xn)
        qkv = ops.transpose(
            ops.reshape(qkv, (B, N, 3, self.num_heads, self.dim_head)), (2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.softmax(
            ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale, axis=-1
        )
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, N, self.inner_dim))
        x = self.to_out(out) + x

        y = self.ff_norm(x)
        y = keras.activations.gelu(self.ff1(y))
        y = self.ff2(y)
        return y + x


class ClayEncoder(keras.Model):

    def __init__(
        self,
        img_size=224,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dim_head=64,
        mlp_ratio=4.0,
        wave_dim=128,
        num_latent_tokens=128,
        name="clay_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid = img_size // patch_size

        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim), initializer="random_normal", name="cls_token"
        )
        self.patch_embedding = DynamicEmbedding(
            wave_dim, num_latent_tokens, patch_size, embed_dim, name="patch_embedding"
        )
        self.blocks = [
            ClayTransformerBlock(
                embed_dim,
                num_heads,
                dim_head,
                int(embed_dim * mlp_ratio),
                name=f"block{i}",
            )
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=_LN_EPS, name="norm")

    def call(self, inputs, gsd=1.0, training=False):
        pixels = inputs["pixels"]
        waves = inputs["waves"]
        time_latlon = inputs["time_latlon"]

        patches, _ = self.patch_embedding(pixels, waves)

        pos = posemb_sincos_2d_with_gsd(
            self.grid, self.grid, self.embed_dim - 8, gsd=gsd
        )
        pos = ops.convert_to_tensor(pos)[None]
        B = ops.shape(patches)[0]
        pos = ops.broadcast_to(pos, (B, ops.shape(pos)[1], self.embed_dim - 8))
        time_latlon = ops.broadcast_to(
            time_latlon[:, None, :], (B, ops.shape(pos)[1], 8)
        )
        patches = patches + ops.concatenate([pos, time_latlon], axis=-1)

        cls = ops.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        tokens = ops.concatenate([cls, patches], axis=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


def ClayClassifier(
    img_size=224,
    patch_size=8,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_bands=10,
    num_classes=10,
    name="clay_classifier",
):
    pixels_in = keras.Input((img_size, img_size, num_bands), name="pixels")
    time_latlon_in = keras.Input((8,), name="time_latlon")
    waves_in = keras.Input((num_bands,), batch_size=1, name="waves")

    encoder = ClayEncoder(
        img_size, patch_size, embed_dim, depth, num_heads, name="encoder"
    )
    tokens = encoder(
        {
            "pixels": pixels_in,
            "waves": layers.Lambda(lambda t: t[0])(waves_in),
            "time_latlon": time_latlon_in,
        }
    )
    cls_token = layers.Lambda(lambda t: t[:, 0], name="take_cls")(tokens)
    outputs = layers.Dense(num_classes, name="head")(cls_token)
    return keras.Model([pixels_in, time_latlon_in, waves_in], outputs, name=name)
