import itertools
import math

import numpy as np
import keras
from keras import layers, ops

_LN_EPS = 1e-5


def get_2d_alibi(num_heads, grid_size):
    num_patches = grid_size * grid_size
    points = list(itertools.product(range(grid_size), range(grid_size)))

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            return [start * start ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2])

    slopes = np.array(get_slopes(num_heads), dtype=np.float64)[:, None]
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = np.concatenate(idxs, axis=1)
    return all_bias.reshape(1, num_heads, num_patches, num_patches).astype(np.float32)


class CromaAttention(layers.Layer):

    def __init__(self, dim, num_heads=16, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def build(self, input_shape):
        self.input_norm = layers.LayerNormalization(epsilon=_LN_EPS, name="input_norm")
        self.to_qkv = layers.Dense(self.dim * 3, use_bias=False, name="to_qkv")
        self.to_out = layers.Dense(self.dim, name="to_out")
        super().build(input_shape)

    def call(self, x, attn_bias):
        x = self.input_norm(x)
        B, N = ops.shape(x)[0], ops.shape(x)[1]
        qkv = self.to_qkv(x)
        qkv = ops.transpose(ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale + attn_bias
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, N, self.dim))
        return self.to_out(out)


class CromaCrossAttention(layers.Layer):

    def __init__(self, dim, num_heads=16, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def build(self, input_shape):
        self.input_norm = layers.LayerNormalization(epsilon=_LN_EPS, name="input_norm")
        self.to_q = layers.Dense(self.dim, use_bias=False, name="to_q")
        self.to_k = layers.Dense(self.dim, use_bias=False, name="to_k")
        self.to_v = layers.Dense(self.dim, use_bias=False, name="to_v")
        self.to_out = layers.Dense(self.dim, name="to_out")
        super().build(input_shape)

    def call(self, x, context, attn_bias):
        xn = self.input_norm(x)
        cn = self.input_norm(context)
        B, N = ops.shape(xn)[0], ops.shape(xn)[1]
        Nc = ops.shape(cn)[1]

        def split_heads(t, n):
            return ops.transpose(ops.reshape(t, (B, n, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        q = split_heads(self.to_q(xn), N)
        k = split_heads(self.to_k(cn), Nc)
        v = split_heads(self.to_v(cn), Nc)

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale + attn_bias
        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, N, self.dim))
        return self.to_out(out)


class CromaFFN(layers.Layer):

    def __init__(self, dim, mult=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.mult = mult

    def build(self, input_shape):
        self.input_norm = layers.LayerNormalization(epsilon=_LN_EPS, name="input_norm")
        self.fc1 = layers.Dense(int(self.dim * self.mult), name="fc1")
        self.fc2 = layers.Dense(self.dim, name="fc2")
        super().build(input_shape)

    def call(self, x):
        x = self.input_norm(x)
        x = keras.activations.gelu(self.fc1(x))
        return self.fc2(x)


class CromaSelfBlock(layers.Layer):

    def __init__(self, dim, num_heads=16, mlp_mult=4, **kwargs):
        super().__init__(**kwargs)
        self.attn = CromaAttention(dim, num_heads, name="attn")
        self.ffn = CromaFFN(dim, mlp_mult, name="ffn")

    def call(self, x, attn_bias):
        x = self.attn(x, attn_bias) + x
        x = self.ffn(x) + x
        return x


class CromaCrossBlock(layers.Layer):

    def __init__(self, dim, num_heads=16, mlp_mult=4, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = CromaAttention(dim, num_heads, name="self_attn")
        self.cross_attn = CromaCrossAttention(dim, num_heads, name="cross_attn")
        self.ffn = CromaFFN(dim, mlp_mult, name="ffn")

    def call(self, x, context, attn_bias):
        x = self.self_attn(x, attn_bias) + x
        x = self.cross_attn(x, context, attn_bias) + x
        x = self.ffn(x) + x
        return x


class ModalityEncoder(keras.Model):

    def __init__(self, dim=768, depth=12, in_chans=2, patch_size=8, num_heads=16,
                 mlp_mult=4, name="modality_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.patch_size = patch_size
        self.linear_input = layers.Dense(dim, name="linear_input")
        self.blocks = [CromaSelfBlock(dim, num_heads, mlp_mult, name=f"block{i}") for i in range(depth)]
        self.norm_out = layers.LayerNormalization(epsilon=_LN_EPS, name="norm_out")

    def call(self, imgs, attn_bias, training=False):
        p = self.patch_size
        B = ops.shape(imgs)[0]
        H, W, C = imgs.shape[1], imgs.shape[2], imgs.shape[3]
        gh, gw = H // p, W // p

        x = ops.reshape(imgs, (B, gh, p, gw, p, C))
        x = ops.transpose(x, (0, 1, 3, 5, 2, 4))
        x = ops.reshape(x, (B, gh * gw, C * p * p))

        x = self.linear_input(x)
        for blk in self.blocks:
            x = blk(x, attn_bias)
        return self.norm_out(x)


def gap_ffn(dim, name):
    return keras.Sequential([
        layers.LayerNormalization(epsilon=_LN_EPS, name="norm"),
        layers.Dense(int(4 * dim), name="fc1"),
        layers.Activation("gelu", name="gelu"),
        layers.Dense(dim, name="fc2"),
    ], name=name)


class CrossAttentionFusion(keras.Model):

    def __init__(self, dim, num_heads=16, depth=6, mlp_mult=4, name="cross_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.blocks = [CromaCrossBlock(dim, num_heads, mlp_mult, name=f"block{i}") for i in range(depth)]
        self.norm_out = layers.LayerNormalization(epsilon=_LN_EPS, name="norm_out")

    def call(self, x, context, attn_bias, training=False):
        for blk in self.blocks:
            x = blk(x, context, attn_bias)
        return self.norm_out(x)


def CROMA(
    img_size=120,
    patch_size=8,
    size="base",
    sar_chans=2,
    optical_chans=12,
    name="croma",
):
    dim, depth = {"base": (768, 12), "large": (1024, 24)}[size]
    num_heads = 16
    grid_size = img_size // patch_size
    attn_bias_np = get_2d_alibi(num_heads, grid_size)

    sar_in = keras.Input((img_size, img_size, sar_chans), name="sar")
    opt_in = keras.Input((img_size, img_size, optical_chans), name="optical")

    attn_bias = ops.convert_to_tensor(attn_bias_np)

    sar_encoder = ModalityEncoder(dim, depth // 2, sar_chans, patch_size, num_heads, name="s1_encoder")
    opt_encoder = ModalityEncoder(dim, depth, optical_chans, patch_size, num_heads, name="s2_encoder")

    sar_tokens = sar_encoder(sar_in, attn_bias)
    opt_tokens = opt_encoder(opt_in, attn_bias)

    fusion = CrossAttentionFusion(dim, num_heads, depth // 2, name="cross_encoder")
    joint_tokens = fusion(sar_tokens, opt_tokens, attn_bias)

    sar_pool = layers.GlobalAveragePooling1D(name="sar_pool")(sar_tokens)
    opt_pool = layers.GlobalAveragePooling1D(name="optical_pool")(opt_tokens)
    sar_repr = gap_ffn(dim, name="GAP_FFN_s1")(sar_pool)
    opt_repr = gap_ffn(dim, name="GAP_FFN_s2")(opt_pool)

    return keras.Model(
        [sar_in, opt_in],
        {"sar_repr": sar_repr, "optical_repr": opt_repr, "joint_tokens": joint_tokens},
        name=name,
    )
