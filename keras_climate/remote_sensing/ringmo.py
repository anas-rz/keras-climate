"""
keras_climate.remote_sensing.ringmo
---------------------------------------
RingMo (Sun et al. 2022): a Swin-Transformer-based masked-image-modeling
foundation model for remote sensing, designed around a masking strategy
("PI-Mask" - Patch Incomplete Mask) that only zeros a fraction of pixels
within each masked block rather than the whole block, better preserving
partial texture/edges of the small, dense objects common in aerial/
satellite imagery than vanilla MAE/SimMIM block masking.

No official checkpoint is publicly downloadable for RingMo: the paper's
own repo (a MindSpore implementation targeting Huawei Ascend hardware) has
never shipped pretrained weights despite multiple long-standing community
requests, and no credible unofficial reproduction redistributes any either.
This module is therefore validated against a from-scratch synthetic
PyTorch reference of its own architecture only (see `test_ringmo.py` and
`weights/mappings/ringmo_mapping.py`), the same tier used elsewhere in this
repo for Clay/AnySat/Earthformer/MetNet/PatchTST/TimesNet/TFT. The exact
patch-embed stem ("pi_conv", a factorized multi-stage conv rather than a
single non-overlapping conv) is also not documented precisely enough in
public sources to reproduce byte-exact even if a checkpoint did exist, so
it is this repo's own reasonable interpretation of the paper's description.
"""

import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import ConvBNAct, MLP


def window_partition(x, window_size):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C). H and W
    must be statically known and divisible by `window_size`."""
    B = ops.shape(x)[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = ops.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (-1, window_size, window_size, C))


def window_reverse(windows, window_size, H, W):
    """Inverse of `window_partition`."""
    C = windows.shape[-1]
    nh, nw = H // window_size, W // window_size
    B = ops.shape(windows)[0] // (nh * nw)
    x = ops.reshape(windows, (B, nh, nw, window_size, window_size, C))
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    return ops.reshape(x, (B, H, W, C))


def pixel_shuffle(x, scale):
    """Depth-to-space: (B, H, W, C*scale^2) -> (B, H*scale, W*scale, C).
    Matches PyTorch's `nn.functional.pixel_shuffle` element-for-element:
    torch decomposes its (NCHW) channel axis as (C, r_h, r_w) - C slowest,
    then r_h, then r_w fastest - so the NHWC channel axis here must
    decompose the same way (out_c, scale, scale), not (scale, scale,
    out_c)."""
    B, H, W, C = ops.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    out_c = C // (scale * scale)
    x = ops.reshape(x, (B, H, W, out_c, scale, scale))
    x = ops.transpose(x, (0, 1, 4, 2, 5, 3))  # (B, H, r_h, W, r_w, out_c)
    return ops.reshape(x, (B, H * scale, W * scale, out_c))


class WindowAttention(layers.Layer):
    """Multi-head self-attention restricted to non-overlapping windows,
    with a learned relative position bias (standard Swin Transformer)."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        coords = np.stack(np.meshgrid(np.arange(window_size), np.arange(window_size), indexing="ij"))
        coords_flat = coords.reshape(2, -1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.transpose(1, 2, 0)
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        self._rel_pos_index_np = rel_coords.sum(-1).astype("int32").reshape(-1)  # (N*N,)

    def build(self, input_shape):
        num_rel = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            shape=(num_rel, self.num_heads), initializer="zeros",
            trainable=True, name="relative_position_bias_table")
        # A plain tensor attribute (rather than `add_weight`) baked in
        # `build()` breaks under the TF backend: `build()` and `call()` can
        # be traced in different FuncGraphs, and a raw constant tensor
        # captured in one graph cannot be referenced from another - unlike
        # a tracked `Variable`, which is graph-agnostic. Same fix as
        # `GSDPositionalEmbedding`'s grid buffers in scalemae.py.
        self.rel_pos_index = self.add_weight(
            shape=self._rel_pos_index_np.shape, dtype="int32",
            initializer=keras.initializers.Constant(self._rel_pos_index_np),
            trainable=False, name="rel_pos_index")
        self.qkv = layers.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")
        self.proj = layers.Dense(self.dim, name="proj")
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (B_, N, C), B_ = num_windows * B
        B_, N, C = ops.shape(x)[0], self.window_size * self.window_size, self.dim
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B_, N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # (B_, heads, N, N)

        bias = ops.take(self.relative_position_bias_table, self.rel_pos_index, axis=0)
        bias = ops.reshape(bias, (N, N, self.num_heads))
        bias = ops.transpose(bias, (2, 0, 1))  # (heads, N, N)
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
    """Pre-norm Swin block: (shifted) window attention + MLP, each with a
    residual connection."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        H, W = input_resolution
        if min(H, W) <= window_size:
            # Feature map no larger than one window: attend over the whole
            # map, no point shifting (mirrors the official edge-case fix
            # for small/late-stage resolutions).
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
            self._build_attn_mask(H, W, window_size, shift_size) if shift_size > 0 else None
        )

    @staticmethod
    def _build_attn_mask(H, W, window_size, shift_size):
        img_mask = np.zeros((1, H, W, 1), dtype="float32")
        slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        m = img_mask.reshape(1, H // window_size, window_size, W // window_size, window_size, 1)
        m = m.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size)
        attn_mask = m[:, None, :] - m[:, :, None]
        return np.where(attn_mask != 0, -100.0, 0.0).astype("float32")

    def build(self, input_shape):
        if self._attn_mask_np is not None:
            # Non-trainable weight, not a plain tensor attribute - see
            # `WindowAttention.build`'s comment on why (TF backend
            # FuncGraph scoping across `build()`/`call()`).
            self.attn_mask = self.add_weight(
                shape=self._attn_mask_np.shape,
                initializer=keras.initializers.Constant(self._attn_mask_np),
                trainable=False, name="attn_mask")
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
    """Concatenates each 2x2 neighborhood of tokens then projects
    4*dim -> 2*dim, halving spatial resolution and doubling channels."""

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
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)
        x = ops.reshape(x, (B, (H // 2) * (W // 2), 4 * C))
        return self.reduction(self.norm(x))


class SwinStage(layers.Layer):
    """One Swin stage: `depth` blocks (alternating regular/shifted windows)
    followed by an optional `PatchMerging` downsample."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, downsample=False, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size,
                                  shift_size=0 if i % 2 == 0 else window_size // 2,
                                  mlp_ratio=mlp_ratio, name=f"block{i}")
            for i in range(depth)
        ]
        self.downsample = PatchMerging(input_resolution, dim, name="downsample") if downsample else None

    def call(self, x, training=False):
        for blk in self.blocks:
            x = blk(x, training=training)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RingMoPatchEmbed(layers.Layer):
    """Factorized 3-stage Conv-BN-GELU stem ("pi_conv") that downsamples by
    4x total (two stride-2 stages), replacing a plain single-conv patch
    embed. This repo's own interpretation of RingMo's stem (see module
    docstring) - only `patch_size=4` is supported."""

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
    """Swin-B-style hierarchical backbone (4 stages, doubling channels and
    halving resolution each stage) for downstream feature extraction.
    Returns the final-stage token sequence `(B, N_final, embed_dim * 8)`."""

    def __init__(self, img_size=192, embed_dim=128, depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32), window_size=6, mlp_ratio=4.0,
                 name="ringmo_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.patch_embed = RingMoPatchEmbed(embed_dim, name="patch_embed")
        grid = img_size // 4
        dim, resolution = embed_dim, grid
        self.stages = []
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            downsample = i < len(depths) - 1
            self.stages.append(SwinStage(dim, (resolution, resolution), depth, heads, window_size,
                                          mlp_ratio, downsample=downsample, name=f"stage{i}"))
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
    """"Patch Incomplete Mask": within each `mask_patch_size` block, only
    `inside_ratio` of pixels are actually zeroed - rather than the whole
    block (vanilla SimMIM/MAE-style masking) - preserving partial texture/
    edges of small dense objects, per RingMo's stated motivation. Has no
    learnable weights; used only for MIM pretraining. Returns
    `(masked_image, mask)`."""

    def __init__(self, mask_patch_size=32, mask_ratio=0.6, inside_ratio=0.6, **kwargs):
        super().__init__(**kwargs)
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio
        self.inside_ratio = inside_ratio

    def call(self, x):
        B, H, W = ops.shape(x)[0], x.shape[1], x.shape[2]
        gh, gw = H // self.mask_patch_size, W // self.mask_patch_size

        block_mask = ops.cast(keras.random.uniform((B, gh, gw)) < self.mask_ratio, "float32")
        pixel_mask = ops.cast(keras.random.uniform((B, H, W)) < self.inside_ratio, "float32")

        block_mask_full = ops.repeat(block_mask, self.mask_patch_size, axis=1)
        block_mask_full = ops.repeat(block_mask_full, self.mask_patch_size, axis=2)
        mask = block_mask_full * pixel_mask
        mask = mask[..., None]
        return x * (1.0 - mask), mask


class SimMIMDecoder(layers.Layer):
    """1x1 conv + pixel-shuffle reconstruction head (SimMIM-style), mapping
    the encoder's final feature grid straight back to full-resolution
    pixels in one upsample."""

    def __init__(self, encoder_stride, in_chans=3, **kwargs):
        super().__init__(**kwargs)
        self.encoder_stride = encoder_stride
        self.in_chans = in_chans

    def build(self, input_shape):
        self.conv = layers.Conv2D(self.in_chans * self.encoder_stride ** 2, 1, name="conv")
        super().build(input_shape)

    def call(self, x):
        return pixel_shuffle(self.conv(x), self.encoder_stride)


def RingMo(img_size=192, in_chans=3, embed_dim=128, depths=(2, 2, 18, 2),
           num_heads=(4, 8, 16, 32), window_size=6, mlp_ratio=4.0,
           mask_patch_size=32, mask_ratio=0.6, inside_ratio=0.6, name="ringmo"):
    """Full MIM pretraining model: returns a `keras.Model` outputting
    `(reconstructed_image, mask)` given an input image."""
    inputs = keras.Input(shape=(img_size, img_size, in_chans), name="image")
    masked, mask = PIMask(mask_patch_size, mask_ratio, inside_ratio, name="pi_mask")(inputs)

    encoder = RingMoEncoder(img_size, embed_dim, depths, num_heads, window_size, mlp_ratio,
                             name=f"{name}_encoder")
    tokens = encoder(masked)

    feat = layers.Reshape((encoder.final_grid, encoder.final_grid, encoder.final_dim),
                           name="reshape_to_grid")(tokens)

    encoder_stride = 4 * (2 ** (len(depths) - 1))
    pred = SimMIMDecoder(encoder_stride, in_chans, name=f"{name}_decoder")(feat)

    return keras.Model(inputs, [pred, mask], name=name)
