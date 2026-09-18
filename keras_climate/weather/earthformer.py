"""
keras_climate.weather.earthformer
-------------------------------------
Earthformer (Gao et al. 2022): a hierarchical spatiotemporal transformer
built from "Cuboid Attention" - self-attention restricted to local
(T x H x W) cuboids of the input tensor, with different cuboid shapes /
strides per layer to capture both local and global structure cheaply.
Encoder-decoder with a UNet-like hierarchy (downsample in space between
encoder stages, upsample + skip connections in the decoder).
"""

import keras
from keras import layers, ops
from ..utils.layers import MLP, DropPath


class CuboidAttention(layers.Layer):
    """Partitions a (B, T, H, W, C) tensor into non-overlapping cuboids of
    shape `cuboid_size = (ct, ch, cw)`, applies self-attention within each
    cuboid independently (batched), then reassembles. `strategy="local"`
    uses contiguous cuboids; `strategy="dilated"` gathers a strided cuboid
    to approximate global context cheaply (a simplified stand-in for the
    paper's local/global cuboid decomposition)."""

    def __init__(self, dim, num_heads, cuboid_size=(2, 4, 4), strategy="local", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cuboid_size = cuboid_size
        self.strategy = strategy

    def build(self, input_shape):
        self.qkv = layers.Dense(self.dim * 3, name="qkv")
        self.proj = layers.Dense(self.dim, name="proj")
        super().build(input_shape)

    def _partition(self, x, T, H, W):
        ct, ch, cw = self.cuboid_size
        ct, ch, cw = min(ct, T), min(ch, H), min(cw, W)
        B, C = ops.shape(x)[0], self.dim
        nt, nh, nw = T // ct, H // ch, W // cw

        if self.strategy == "local":
            x = ops.reshape(x, (B, nt, ct, nh, ch, nw, cw, C))
            x = ops.transpose(x, (0, 1, 3, 5, 2, 4, 6, 7))  # B nt nh nw ct ch cw C
        else:  # "dilated": swap the roles of (n, c) to gather strided elements
            x = ops.reshape(x, (B, ct, nt, ch, nh, cw, nw, C))
            x = ops.transpose(x, (0, 2, 4, 6, 1, 3, 5, 7))  # B nt nh nw ct ch cw C

        num_cuboids = nt * nh * nw
        cuboid_len = ct * ch * cw
        x = ops.reshape(x, (B * num_cuboids, cuboid_len, C))
        return x, (B, nt, nh, nw, ct, ch, cw, C)

    def _reassemble(self, x, meta):
        B, nt, nh, nw, ct, ch, cw, C = meta
        x = ops.reshape(x, (B, nt, nh, nw, ct, ch, cw, C))
        if self.strategy == "local":
            x = ops.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))  # B nt ct nh ch nw cw C
        else:
            x = ops.transpose(x, (0, 4, 1, 5, 2, 6, 3, 7))  # B ct nt ch nh cw nw C
        x = ops.reshape(x, (B, nt * ct, nh * ch, nw * cw, C))
        return x

    def call(self, x, training=False):
        T, H, W = ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]
        cuboids, meta = self._partition(x, T, H, W)

        qkv = self.qkv(cuboids)
        N = ops.shape(cuboids)[1]
        qkv = ops.reshape(qkv, (ops.shape(cuboids)[0], N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ops.softmax(ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (ops.shape(cuboids)[0], N, self.dim))
        out = self.proj(out)

        return self._reassemble(out, meta)


class CuboidTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, cuboid_size=(2, 4, 4), strategy="local",
                 mlp_ratio=4.0, drop_path=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attn = CuboidAttention(dim, num_heads, cuboid_size, strategy, name="attn")
        self.drop_path = DropPath(drop_path)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.mlp = MLP(int(dim * mlp_ratio), dim, name="mlp")

    def call(self, x, training=False):
        x = x + self.drop_path(self.attn(self.norm1(x), training=training), training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x), training=training), training=training)
        return x


def _stage(x, dim, num_heads, depth, cuboid_sizes, name):
    for i in range(depth):
        strategy = "local" if i % 2 == 0 else "dilated"
        x = CuboidTransformerBlock(dim, num_heads, cuboid_sizes[i % len(cuboid_sizes)],
                                    strategy=strategy, name=f"{name}_block{i}")(x)
    return x


def Earthformer(
    input_shape=(10, 128, 128, 1),  # (T_in, H, W, C)
    pred_steps=10,
    base_dim=64,
    stage_depths=(2, 2, 2),
    num_heads=4,
    cuboid_size=(2, 4, 4),
    name="earthformer",
):
    """Hierarchical encoder-decoder over (T, H, W). Spatial resolution is
    halved between encoder stages (via strided conv) and doubled back in the
    decoder (via transposed conv), with skip connections, in the spirit of
    Earthformer's UNet-style Cuboid Transformer backbone. The output channel
    is projected to `pred_steps` future frames."""
    T_in, H, W, C_in = input_shape
    inputs = keras.Input(shape=input_shape, name="frames")

    x = layers.TimeDistributed(layers.Conv2D(base_dim, 3, padding="same"), name="stem")(inputs)

    skips = []
    dim = base_dim
    for stage_i, depth in enumerate(stage_depths):
        x = _stage(x, dim, num_heads, depth, [cuboid_size], name=f"enc_stage{stage_i}")
        skips.append(x)
        if stage_i < len(stage_depths) - 1:
            x = layers.TimeDistributed(layers.Conv2D(dim * 2, 3, strides=2, padding="same"),
                                        name=f"downsample{stage_i}")(x)
            dim *= 2

    for stage_i in reversed(range(len(stage_depths) - 1)):
        dim //= 2
        x = layers.TimeDistributed(layers.Conv2DTranspose(dim, 3, strides=2, padding="same"),
                                    name=f"upsample{stage_i}")(x)
        x = layers.Concatenate(axis=-1, name=f"skip_concat{stage_i}")([x, skips[stage_i]])
        x = layers.TimeDistributed(layers.Conv2D(dim, 1), name=f"skip_proj{stage_i}")(x)
        x = _stage(x, dim, num_heads, stage_depths[stage_i], [cuboid_size], name=f"dec_stage{stage_i}")

    # Project the T_in encoded frames to pred_steps output frames along time,
    # then a pointwise conv to the target channel count.
    x = layers.Permute((2, 3, 1, 4), name="move_time_last")(x)  # B H W T C
    x = layers.Reshape((H, W, T_in * dim), name="merge_time_channel")(x)
    x = layers.Dense(pred_steps * C_in, name="time_channel_proj")(x)
    x = layers.Reshape((H, W, pred_steps, C_in), name="split_time_channel")(x)
    out = layers.Permute((3, 1, 2, 4), name="move_time_first")(x)

    return keras.Model(inputs, out, name=name)
