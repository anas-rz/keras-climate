"""
keras_climate.utils.layers
---------------------------
Shared building blocks used across remote_sensing / forecasting / weather /
foundation model families. Written against Keras 3 (backend-agnostic:
TensorFlow, JAX or PyTorch backend all work).
"""

import math
import numpy as np
import keras
from keras import layers, ops


# --------------------------------------------------------------------------
# Generic utility layers
# --------------------------------------------------------------------------

class DropPath(layers.Layer):
    """Stochastic depth, per-sample path dropout (as used in ViT/Swin/ConvNeXt)."""

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + keras.random.uniform(shape, dtype=x.dtype)
        random_tensor = ops.floor(random_tensor)
        return (x / keep_prob) * random_tensor

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg


class MLP(layers.Layer):
    """Standard transformer MLP (fc -> act -> drop -> fc -> drop)."""

    def __init__(self, hidden_dim, out_dim=None, act="gelu", drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.act = act
        self.drop = drop

    def build(self, input_shape):
        out_dim = self.out_dim or input_shape[-1]
        self.fc1 = layers.Dense(self.hidden_dim, name="fc1")
        self.act_fn = layers.Activation(self.act)
        self.fc2 = layers.Dense(out_dim, name="fc2")
        self.drop1 = layers.Dropout(self.drop)
        self.drop2 = layers.Dropout(self.drop)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class GatedResidualNetwork(layers.Layer):
    """GRN as used in TFT: nonlinear layer with a gated skip connection."""

    def __init__(self, units, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout

    def build(self, input_shape):
        # Explicit names are required, not cosmetic: without them Keras
        # auto-assigns globally-incrementing names ("dense_16", ...) that
        # depend on how many other unnamed Dense/LayerNormalization layers
        # were created earlier in the same process - since TFT builds many
        # GRN instances, this makes weight-porting by name unreproducible.
        self.skip = layers.Dense(self.units, name="skip") if input_shape[-1] != self.units else None
        self.fc1 = layers.Dense(self.units, activation="elu", name="fc1")
        self.fc2 = layers.Dense(self.units, name="fc2")
        self.drop = layers.Dropout(self.dropout)
        self.gate = layers.Dense(self.units * 2, name="gate")
        self.norm = layers.LayerNormalization(name="norm")
        super().build(input_shape)

    def call(self, x, training=False):
        skip = self.skip(x) if self.skip is not None else x
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.drop(h, training=training)
        gated = self.gate(h)
        value, gate = ops.split(gated, 2, axis=-1)
        h = value * ops.sigmoid(gate)
        return self.norm(skip + h)


# --------------------------------------------------------------------------
# Attention blocks
# --------------------------------------------------------------------------

class MultiHeadSelfAttention(layers.Layer):
    """Standard ViT-style multi-head self attention."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop

    def build(self, input_shape):
        self.qkv = layers.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")
        self.attn_drop = layers.Dropout(self.attn_drop_rate)
        self.proj = layers.Dense(self.dim, name="proj")
        self.proj_drop = layers.Dropout(self.proj_drop_rate)
        super().build(input_shape)

    def call(self, x, training=False):
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], self.dim
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B, N, C))
        out = self.proj(out)
        out = self.proj_drop(out, training=training)
        return out


class SpatialReductionAttention(layers.Layer):
    """Efficient self-attention with spatial reduction (SegFormer / PVT)."""

    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop

    def build(self, input_shape):
        self.q = layers.Dense(self.dim, use_bias=self.qkv_bias, name="q")
        self.kv = layers.Dense(self.dim * 2, use_bias=self.qkv_bias, name="kv")
        if self.sr_ratio > 1:
            self.sr = layers.Conv2D(self.dim, kernel_size=self.sr_ratio,
                                     strides=self.sr_ratio, name="sr")
            self.norm = layers.LayerNormalization(name="sr_norm")
        self.attn_drop = layers.Dropout(self.attn_drop_rate)
        self.proj = layers.Dense(self.dim, name="proj")
        self.proj_drop = layers.Dropout(self.proj_drop_rate)
        super().build(input_shape)

    def call(self, x, H, W, training=False):
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], self.dim
        q = self.q(x)
        q = ops.reshape(q, (B, N, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = ops.reshape(x, (B, H, W, C))
            x_ = self.sr(x_)
            new_hw = ops.shape(x_)[1] * ops.shape(x_)[2]
            x_ = ops.reshape(x_, (B, new_hw, C))
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_)
        n2 = ops.shape(x_)[1]
        kv = ops.reshape(kv, (B, n2, 2, self.num_heads, self.head_dim))
        kv = ops.transpose(kv, (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B, N, C))
        out = self.proj(out)
        out = self.proj_drop(out, training=training)
        return out


class TransformerEncoderBlock(layers.Layer):
    """Pre-norm ViT encoder block: LN -> MHSA -> +res -> LN -> MLP -> +res."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, attn_drop, drop, name="attn")
        self.drop_path = DropPath(drop_path)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.mlp = MLP(int(dim * mlp_ratio), dim, drop=drop, name="mlp")

    def call(self, x, training=False):
        x = x + self.drop_path(self.attn(self.norm1(x), training=training), training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x), training=training), training=training)
        return x


class SegformerBlock(layers.Layer):
    """MiT (SegFormer backbone) encoder block using spatial-reduction attention
    and a depthwise-conv "Mix-FFN" instead of a plain MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, sr_ratio=1, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.attn = SpatialReductionAttention(dim, num_heads, sr_ratio, attn_drop=attn_drop,
                                               proj_drop=drop, name="attn")
        self.drop_path = DropPath(drop_path)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        hidden = int(dim * mlp_ratio)
        self.fc1 = layers.Dense(hidden, name="mixffn_fc1")
        self.dwconv = layers.DepthwiseConv2D(3, padding="same", name="mixffn_dwconv")
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(dim, name="mixffn_fc2")
        self.drop = layers.Dropout(drop)

    def call(self, x, H, W, training=False):
        x = x + self.drop_path(self.attn(self.norm1(x), H=H, W=W, training=training), training=training)

        y = self.norm2(x)
        y = self.fc1(y)
        B, N, C = ops.shape(y)[0], ops.shape(y)[1], y.shape[-1]
        y = ops.reshape(y, (B, H, W, C))
        y = self.dwconv(y)
        y = ops.reshape(y, (B, N, C))
        y = self.act(y)
        y = self.drop(y, training=training)
        y = self.fc2(y)
        y = self.drop(y, training=training)
        x = x + self.drop_path(y, training=training)
        return x


# --------------------------------------------------------------------------
# Patch embedding
# --------------------------------------------------------------------------

class PatchEmbed2D(layers.Layer):
    """Conv-based non-overlapping patch embedding, ViT-style. Input NHWC -> (B, N, C)."""

    def __init__(self, patch_size, embed_dim, in_chans=None, norm=True, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_norm = norm

    def build(self, input_shape):
        self.proj = layers.Conv2D(self.embed_dim, kernel_size=self.patch_size,
                                   strides=self.patch_size, name="proj")
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm") if self.use_norm else None
        super().build(input_shape)

    def call(self, x):
        x = self.proj(x)
        H, W = ops.shape(x)[1], ops.shape(x)[2]
        x = ops.reshape(x, (ops.shape(x)[0], H * W, self.embed_dim))
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W


class OverlapPatchEmbed(layers.Layer):
    """Overlapping-stride patch embedding used in SegFormer/PVTv2 stages."""

    def __init__(self, patch_size, stride, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # Same rationale as `ConvBNAct`: this is a strided conv, so Keras's
        # "same" padding can pad asymmetrically while the official SegFormer
        # implementation always pads symmetrically with `patch_size // 2` -
        # pad explicitly + "valid" to match it exactly (needed for faithful
        # weight porting from a real MiT checkpoint).
        pad = self.patch_size // 2
        self.pad = layers.ZeroPadding2D(pad, name="pad") if pad > 0 else None
        self.proj = layers.Conv2D(self.embed_dim, kernel_size=self.patch_size,
                                   strides=self.stride, padding="valid", name="proj")
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        super().build(input_shape)

    def call(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = self.proj(x)
        H, W = ops.shape(x)[1], ops.shape(x)[2]
        x = ops.reshape(x, (ops.shape(x)[0], H * W, self.embed_dim))
        x = self.norm(x)
        return x, H, W


class PatchEmbed3D(layers.Layer):
    """Spatiotemporal patch embedding for video / multi-temporal satellite stacks.
    Input: (B, T, H, W, C) -> tokens (B, N, D). Used by SatMAE-temporal / Prithvi."""

    def __init__(self, patch_size, tubelet_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.proj = layers.Conv3D(
            self.embed_dim,
            kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
            strides=(self.tubelet_size, self.patch_size, self.patch_size),
            name="proj",
        )
        super().build(input_shape)

    def call(self, x):
        x = self.proj(x)
        T, H, W = ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]
        x = ops.reshape(x, (ops.shape(x)[0], T * H * W, self.embed_dim))
        return x, T, H, W


def sincos_position_embedding(length, dim):
    """Fixed (non-learned) 1D sin-cos positional embedding, numpy, shape (length, dim)."""
    position = np.arange(length)[:, None]
    div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    pe = np.zeros((length, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def sincos_position_embedding_2d(h, w, dim):
    """Fixed 2D sin-cos positional embedding (ViT-MAE style), shape (h*w, dim)."""
    assert dim % 2 == 0
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape(2, 1, h, w)

    def embed_1d(pos, d):
        omega = np.arange(d // 2, dtype=np.float32) / (d / 2.0)
        omega = 1.0 / (10000 ** omega)
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = embed_1d(grid[0], dim // 2)
    emb_w = embed_1d(grid[1], dim // 2)
    return np.concatenate([emb_h, emb_w], axis=1)


# --------------------------------------------------------------------------
# Conv building blocks (remote sensing / weather CNNs)
# --------------------------------------------------------------------------

class ConvBNAct(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, dilation_rate=1,
                 act="relu", use_bn=True, **kwargs):
        super().__init__(**kwargs)
        # Keras's `padding="same"` pads *asymmetrically* (extra on the
        # bottom/right) whenever `strides > 1` and the input size isn't an
        # exact multiple of the stride in a way that makes the needed
        # padding even - PyTorch reference convs almost always pad
        # symmetrically via an explicit `padding=dilation*(k-1)//2`, so
        # relying on "same" here would silently misalign every ported
        # strided conv (stem/downsampling convs) against its source
        # checkpoint. Pad explicitly + "valid" whenever strides > 1 to
        # reproduce PyTorch's convention exactly; for strides == 1, "same"
        # is already symmetric, so there's nothing to fix.
        if strides > 1:
            pad = dilation_rate * (kernel_size - 1) // 2
            self.pad = layers.ZeroPadding2D(pad, name="pad") if pad > 0 else None
            conv_padding = "valid"
        else:
            self.pad = None
            conv_padding = "same"
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding=conv_padding,
                                   dilation_rate=dilation_rate, use_bias=not use_bn, name="conv")
        # epsilon=1e-5 matches PyTorch's `nn.BatchNorm2d` default (Keras's
        # own default is 1e-3), which matters for numerically-faithful
        # weight porting from the PyTorch reference implementations this
        # layer is meant to receive checkpoints from.
        self.bn = layers.BatchNormalization(epsilon=1e-5, name="bn") if use_bn else None
        self.act = layers.Activation(act) if act else None

    def call(self, x, training=False):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x


class DoubleConv(layers.Layer):
    """Two 3x3 conv-bn-relu blocks, the basic UNet unit."""

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = ConvBNAct(filters, 3, name="conv1")
        self.conv2 = ConvBNAct(filters, 3, name="conv2")

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class ASPP(layers.Layer):
    """Atrous Spatial Pyramid Pooling head, as used in DeepLabV3(+)."""

    def __init__(self, filters=256, rates=(6, 12, 18), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.rates = rates

    def build(self, input_shape):
        self.branch1 = ConvBNAct(self.filters, 1, name="b0")
        self.branches = [ConvBNAct(self.filters, 3, dilation_rate=r, name=f"b{r}")
                          for r in self.rates]
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.pool_conv = ConvBNAct(self.filters, 1, name="pool_conv")
        self.project = ConvBNAct(self.filters, 1, name="project")
        self.dropout = layers.Dropout(0.1)
        super().build(input_shape)

    def call(self, x, training=False):
        h, w = ops.shape(x)[1], ops.shape(x)[2]
        feats = [self.branch1(x, training=training)]
        feats += [b(x, training=training) for b in self.branches]
        pooled = self.pool(x)
        pooled = self.pool_conv(pooled, training=training)
        pooled = ops.image.resize(pooled, (h, w), interpolation="bilinear")
        feats.append(pooled)
        x = ops.concatenate(feats, axis=-1)
        x = self.project(x, training=training)
        x = self.dropout(x, training=training)
        return x
