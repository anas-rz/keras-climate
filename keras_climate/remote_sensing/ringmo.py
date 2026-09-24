import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import ConvBNAct, MLP


def window_partition(x, window_size):
    B = ops.shape(x)[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = ops.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, window_size, window_size, C))


def window_reverse(windows, window_size, H, W):
    C = windows.shape[-1]
    nh, nw = H // window_size, W // window_size
    B = ops.shape(windows)[0] // (nh * nw)
    x = ops.reshape(windows, (B, nh, nw, window_size, window_size, C))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (B, H, W, C))


def pixel_shuffle(x, scale):
    B, H, W, C = ops.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    out_c = C // (scale * scale)
    x = ops.reshape(x, (B, H, W, out_c, scale, scale))
    x = ops.transpose(x, (0, 1, 4, 2, 5, 3))
    return ops.reshape(x, (B, H * scale, W * scale, out_c))


class WindowAttention(layers.Layer):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias

        coords = np.stack(
            np.meshgrid(np.arange(window_size), np.arange(window_size), indexing="ij")
        )
        coords_flat = coords.reshape(2, -1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.transpose(1, 2, 0)
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        self._rel_pos_index_np = rel_coords.sum(-1).astype("int32").reshape(-1)

    def build(self, input_shape):
        num_rel = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            shape=(num_rel, self.num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        self.rel_pos_index = self.add_weight(
            shape=self._rel_pos_index_np.shape,
            dtype="int32",
            initializer=keras.initializers.Constant(self._rel_pos_index_np),
            trainable=False,
            name="rel_pos_index",
        )
        self.qkv = layers.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")
        self.proj = layers.Dense(self.dim, name="proj")
        super().build(input_shape)

    def call(self, x, mask=None):
        B_, N, C = ops.shape(x)[0], self.window_size * self.window_size, self.dim
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B_, N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))

        bias = ops.take(self.relative_position_bias_table, self.rel_pos_index, axis=0)
        bias = ops.reshape(bias, (N, N, self.num_heads))
        bias = ops.transpose(bias, (2, 0, 1))
        attn = attn + bias[None]

        if mask is not None:
            nw = ops.shape(mask)[0]
            attn = ops.reshape(attn, (B_ // nw, nw, self.num_heads, N, N))
            attn = attn + mask[None, :, None, :, :]
            attn = ops.reshape(attn, (B_, self.num_heads, N, N))

        attn = ops.softmax(attn, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B_, N, C))
        return self.proj(out)


class SwinTransformerBlock(layers.Layer):

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        H, W = input_resolution
        if min(H, W) <= window_size:
            shift_size = 0
            window_size = min(H, W)
        self.input_resolution = (H, W)
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, name="attn")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.mlp = MLP(int(dim * mlp_ratio), dim, name="mlp")

        self._attn_mask_np = (
            self._build_attn_mask(H, W, window_size, shift_size)
            if shift_size > 0
            else None
        )

    @staticmethod
    def _build_attn_mask(H, W, window_size, shift_size):
        img_mask = np.zeros((1, H, W, 1), dtype="float32")
        slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        m = img_mask.reshape(
            1, H // window_size, window_size, W // window_size, window_size, 1
        )
        m = m.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size)
        attn_mask = m[:, None, :] - m[:, :, None]
        return np.where(attn_mask != 0, -100.0, 0.0).astype("float32")

    def build(self, input_shape):
        if self._attn_mask_np is not None:
            self.attn_mask = self.add_weight(
                shape=self._attn_mask_np.shape,
                initializer=keras.initializers.Constant(self._attn_mask_np),
                trainable=False,
                name="attn_mask",
            )
        else:
            self.attn_mask = None
        super().build(input_shape)

    def call(self, x, training=False):
        H, W = self.input_resolution
        B, C = ops.shape(x)[0], self.dim
        shortcut = x
        x = self.norm1(x)
        x = ops.reshape(x, (B, H, W, C))

        if self.shift_size > 0:
            x = ops.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))

        windows = window_partition(x, self.window_size)
        windows = ops.reshape(windows, (-1, self.window_size * self.window_size, C))
        attn_out = self.attn(windows, mask=self.attn_mask)
        attn_out = ops.reshape(attn_out, (-1, self.window_size, self.window_size, C))
        x = window_reverse(attn_out, self.window_size, H, W)

        if self.shift_size > 0:
            x = ops.roll(x, shift=(self.shift_size, self.shift_size), axis=(1, 2))

        x = ops.reshape(x, (B, H * W, C))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x), training=training)
        return x


class PatchMerging(layers.Layer):

    def __init__(self, input_resolution, dim, **kwargs):
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.reduction = layers.Dense(2 * dim, use_bias=False, name="reduction")

    def call(self, x):
        H, W = self.input_resolution
        B, C = ops.shape(x)[0], self.dim
        x = ops.reshape(x, (B, H, W, C))
        x0, x1, x2, x3 = (
            x[:, 0::2, 0::2, :],
            x[:, 1::2, 0::2, :],
            x[:, 0::2, 1::2, :],
            x[:, 1::2, 1::2, :],
        )
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)
        x = ops.reshape(x, (B, (H // 2) * (W // 2), 4 * C))
        return self.reduction(self.norm(x))


class SwinStage(layers.Layer):

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        downsample=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.blocks = [
            SwinTransformerBlock(
                dim,
                input_resolution,
                num_heads,
                window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                name=f"block{i}",
            )
            for i in range(depth)
        ]
        self.downsample = (
            PatchMerging(input_resolution, dim, name="downsample")
            if downsample
            else None
        )

    def call(self, x, training=False):
        for blk in self.blocks:
            x = blk(x, training=training)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RingMoPatchEmbed(layers.Layer):

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        mid = self.embed_dim // 2
        self.conv1 = ConvBNAct(mid, 3, strides=2, act="gelu", name="conv1")
        self.conv2 = ConvBNAct(mid, 3, strides=2, act="gelu", name="conv2")
        self.conv3 = layers.Conv2D(self.embed_dim, 1, name="conv3")
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x)
        H, W = ops.shape(x)[1], ops.shape(x)[2]
        x = ops.reshape(x, (ops.shape(x)[0], H * W, self.embed_dim))
        x = self.norm(x)
        return x, H, W


class RingMoEncoder(keras.Model):

    def __init__(
        self,
        img_size=192,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=6,
        mlp_ratio=4.0,
        name="ringmo_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.patch_embed = RingMoPatchEmbed(embed_dim, name="patch_embed")
        grid = img_size // 4
        dim, resolution = embed_dim, grid
        self.stages = []
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            downsample = i < len(depths) - 1
            self.stages.append(
                SwinStage(
                    dim,
                    (resolution, resolution),
                    depth,
                    heads,
                    window_size,
                    mlp_ratio,
                    downsample=downsample,
                    name=f"stage{i}",
                )
            )
            if downsample:
                dim *= 2
                resolution //= 2
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.final_dim = dim
        self.final_grid = resolution

    def call(self, x, training=False):
        tokens, H, W = self.patch_embed(x, training=training)
        for stage in self.stages:
            tokens = stage(tokens, training=training)
        return self.norm(tokens)


class PIMask(layers.Layer):

    def __init__(self, mask_patch_size=32, mask_ratio=0.6, inside_ratio=0.6, **kwargs):
        super().__init__(**kwargs)
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        self.inside_ratio = inside_ratio
        self.seed_generator = keras.random.SeedGenerator()

    def call(self, x):
        B, H, W = ops.shape(x)[0], x.shape[1], x.shape[2]
        gh, gw = H // self.mask_patch_size, W // self.mask_patch_size

        block_mask = ops.cast(
            keras.random.uniform((B, gh, gw), seed=self.seed_generator) < self.mask_ratio,
            "float32",
        )
        pixel_mask = ops.cast(
            keras.random.uniform((B, H, W), seed=self.seed_generator) < self.inside_ratio,
            "float32",
        )

        block_mask_full = ops.repeat(block_mask, self.mask_patch_size, axis=1)
        block_mask_full = ops.repeat(block_mask_full, self.mask_patch_size, axis=2)
        mask = block_mask_full * pixel_mask
        mask = mask[..., None]
        return x * (1.0 - mask), mask


class SimMIMDecoder(layers.Layer):

    def __init__(self, encoder_stride, in_chans=3, **kwargs):
        super().__init__(**kwargs)
        self.encoder_stride = encoder_stride
        self.in_chans = in_chans

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            self.in_chans * self.encoder_stride**2, 1, name="conv"
        )
        super().build(input_shape)

    def call(self, x):
        return pixel_shuffle(self.conv(x), self.encoder_stride)


def RingMo(
    img_size=192,
    in_chans=3,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    window_size=6,
    mlp_ratio=4.0,
    mask_patch_size=32,
    mask_ratio=0.6,
    inside_ratio=0.6,
    name="ringmo",
):
    inputs = keras.Input(shape=(img_size, img_size, in_chans), name="image")
    masked, mask = PIMask(mask_patch_size, mask_ratio, inside_ratio, name="pi_mask")(
        inputs
    )

    encoder = RingMoEncoder(
        img_size,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio,
        name=f"{name}_encoder",
    )
    tokens = encoder(masked)

    feat = layers.Reshape(
        (encoder.final_grid, encoder.final_grid, encoder.final_dim),
        name="reshape_to_grid",
    )(tokens)

    encoder_stride = 4 * (2 ** (len(depths) - 1))
    pred = SimMIMDecoder(encoder_stride, in_chans, name=f"{name}_decoder")(feat)

    return keras.Model(inputs, [pred, mask], name=name)
