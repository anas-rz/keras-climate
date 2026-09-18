"""
keras_climate.foundation.croma
-----------------------------------
CROMA (Fuller et al. 2023): a dual-encoder foundation model that jointly
represents SAR (e.g. Sentinel-1) and optical (e.g. Sentinel-2) imagery.
Each modality has its own ViT encoder; a cross-attention fusion stack
produces a joint multimodal representation. Pretraining combines a
contrastive objective (aligning the two modalities' global representations)
with per-modality masked-image modeling; this module exposes the encoders
and fusion block so either objective can be wired up externally, plus a
convenience `CROMA(...)` model that returns unimodal + joint embeddings.

This is a faithful port of the official implementation
(github.com/antofuller/CROMA, `use_croma.py`), not just an approximation -
matching it exactly matters here because there is a real, publicly
released checkpoint (`antofuller/CROMA` on HuggingFace) with three notable
architectural choices that are easy to get wrong by "reasonable-sounding"
default assumptions:

  * Patch embedding is a plain `Linear` over flattened raw pixel patches
    (`(c, patch_h, patch_w) -> dim`), not a Conv2D-based patch embedding.
  * There is no learned position embedding at all - instead, every
    attention layer adds a fixed (non-trainable) 2D ALiBi bias, computed
    once from pairwise patch-grid distances.
  * The SAR encoder is intentionally *half* the depth of the optical
    encoder, and the joint/cross encoder is a single query stream (SAR)
    progressively self- and cross-attending against a *fixed* optical
    context - not a bidirectional update of both streams.
"""

import itertools
import math

import numpy as np
import keras
from keras import layers, ops

# PyTorch's plain `nn.LayerNorm(dim)` (no eps override, as used throughout
# the reference) defaults to eps=1e-5 - Keras's own default is 1e-3, and
# this codebase's other ViT blocks use 1e-6 (matching timm/MAE), so this
# must be set explicitly to match the CROMA checkpoint.
_LN_EPS = 1e-5


def get_2d_alibi(num_heads, grid_size):
    """Fixed (non-learned) 2D ALiBi attention bias, shape
    (1, num_heads, num_patches, num_patches) - numpy port of the official
    `get_2dalibi`, matching it bit-for-bit (pure function of grid geometry,
    no learned parameters, so this never needs to be "ported" - just
    recomputed identically)."""
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

    slopes = np.array(get_slopes(num_heads), dtype=np.float64)[:, None]  # (num_heads, 1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = np.concatenate(idxs, axis=1)  # (num_heads, num_patches*num_patches)
    return all_bias.reshape(1, num_heads, num_patches, num_patches).astype(np.float32)


class CromaAttention(layers.Layer):
    """Self-attention with an additive (fixed) relative-position bias and
    a bias-free fused QKV projection - matches the reference `Attention`."""

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
    """Cross-attention: query stream `x` attends to a separate `context`
    stream. The reference applies the *same* `input_norm` module to both
    `x` and `context` (shared weights) - reusing one Keras layer instance
    for both calls reproduces that exactly."""

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
    """Pre-norm MLP: `input_norm -> Linear -> GELU -> Linear`."""

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
    """One `BaseTransformer` layer: self-attention + FFN, each with its own
    pre-residual (`x = sublayer(x) + x`, not `x = x + sublayer(norm(x))`
    factored externally - the norm lives inside each sublayer)."""

    def __init__(self, dim, num_heads=16, mlp_mult=4, **kwargs):
        super().__init__(**kwargs)
        self.attn = CromaAttention(dim, num_heads, name="attn")
        self.ffn = CromaFFN(dim, mlp_mult, name="ffn")

    def call(self, x, attn_bias):
        x = self.attn(x, attn_bias) + x
        x = self.ffn(x) + x
        return x


class CromaCrossBlock(layers.Layer):
    """One `BaseTransformerCrossAttn` layer: self-attn, then cross-attn
    against a fixed `context`, then FFN."""

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
    """A standard ViT encoder for one modality (SAR or optical): flatten
    non-overlapping pixel patches, project with a single `Dense` (matching
    the reference's `Linear` patch embedding exactly - not a Conv2D), then
    a stack of `CromaSelfBlock`s using a fixed 2D-ALiBi attention bias
    (`attn_bias` is passed in at call time, since it depends on grid_size,
    not per-encoder state)."""

    def __init__(self, dim=768, depth=12, in_chans=2, patch_size=8, num_heads=16,
                 mlp_mult=4, name="modality_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.patch_size = patch_size
        self.linear_input = layers.Dense(dim, name="linear_input")
        self.blocks = [CromaSelfBlock(dim, num_heads, mlp_mult, name=f"block{i}") for i in range(depth)]
        self.norm_out = layers.LayerNormalization(epsilon=_LN_EPS, name="norm_out")

    def call(self, imgs, attn_bias, training=False):
        # imgs: (B, H, W, C) - Keras NHWC convention (reference is NCHW).
        p = self.patch_size
        B = ops.shape(imgs)[0]
        H, W, C = imgs.shape[1], imgs.shape[2], imgs.shape[3]
        gh, gw = H // p, W // p

        # Reproduce the reference's exact per-patch flatten order,
        # `'b c (h i) (w j) -> b (h w) (c i j)'` (channel slowest, then
        # patch-row, then patch-col), despite our channels-last input.
        x = ops.reshape(imgs, (B, gh, p, gw, p, C))
        x = ops.transpose(x, (0, 1, 3, 5, 2, 4))  # (B, gh, gw, C, p, p)
        x = ops.reshape(x, (B, gh * gw, C * p * p))

        x = self.linear_input(x)
        for blk in self.blocks:
            x = blk(x, attn_bias)
        return self.norm_out(x)


def gap_ffn(dim, name):
    """`GAP_FFN_{s1,s2}`: LayerNorm -> Linear(dim, 4*dim) -> GELU ->
    Linear(4*dim, dim), applied to the mean-pooled encoder output."""
    return keras.Sequential([
        layers.LayerNormalization(epsilon=_LN_EPS, name="norm"),
        layers.Dense(int(4 * dim), name="fc1"),
        layers.Activation("gelu", name="gelu"),
        layers.Dense(dim, name="fc2"),
    ], name=name)


class CrossAttentionFusion(keras.Model):
    """`BaseTransformerCrossAttn`: a single query stream (SAR) that
    self-attends, then cross-attends against a *fixed* context (optical),
    then FFN - repeated `depth` times - followed by a final LayerNorm.
    Output length matches the query stream, not a concatenation of both."""

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
    """Returns a model mapping {sar, optical} -> {sar_repr, optical_repr,
    joint_repr}, where sar_repr/optical_repr are the GAP_FFN-projected
    mean-pooled unimodal embeddings (for the contrastive objective) and
    joint_repr is the fused multimodal token sequence (for downstream
    dense/fusion tasks). `size="base"` (dim=768, depth=12) or `"large"`
    (dim=1024, depth=24) matches the two officially released checkpoints;
    both always use 16 attention heads and patch_size=8 (hardcoded in the
    reference regardless of what's passed to it)."""
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
