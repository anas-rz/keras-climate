"""
keras_climate.weather.metnet
---------------------------------
MetNet / MetNet-2 style (Sønderby et al. 2020; Espeholt et al. 2022):
a convolutional context-aggregating encoder (large receptive field via
dilated/strided convs) followed by temporal encoding across input frames
and axial self-attention across space (row-wise then column-wise, which is
far cheaper than full spatial self-attention on high-res radar fields),
producing per-lead-time precipitation probability maps.
"""

import keras
from keras import layers, ops


class AxialAttention2D(layers.Layer):
    """Applies self-attention along rows, then along columns, of a
    (B, H, W, C) feature map - linear-ish cost approximation of full 2D
    spatial self-attention, as used in MetNet's spatial aggregator."""

    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def build(self, input_shape):
        self.row_qkv = layers.Dense(self.dim * 3, name="row_qkv")
        self.row_proj = layers.Dense(self.dim, name="row_proj")
        self.col_qkv = layers.Dense(self.dim * 3, name="col_qkv")
        self.col_proj = layers.Dense(self.dim, name="col_proj")
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        super().build(input_shape)

    def _attend(self, x, qkv_layer, proj_layer):
        # x: (B*, L, C) -- generic self-attention along axis 1
        B_, L, C = ops.shape(x)[0], ops.shape(x)[1], self.dim
        qkv = qkv_layer(x)
        qkv = ops.reshape(qkv, (B_, L, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.softmax(ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B_, L, C))
        return proj_layer(out)

    def call(self, x, training=False):
        B, H, W, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], self.dim

        y = self.norm1(x)
        rows = ops.reshape(y, (B * H, W, C))
        rows = self._attend(rows, self.row_qkv, self.row_proj)
        rows = ops.reshape(rows, (B, H, W, C))
        x = x + rows

        y = self.norm2(x)
        cols = ops.transpose(y, (0, 2, 1, 3))
        cols = ops.reshape(cols, (B * W, H, C))
        cols = self._attend(cols, self.col_qkv, self.col_proj)
        cols = ops.reshape(cols, (B, W, H, C))
        cols = ops.transpose(cols, (0, 2, 1, 3))
        x = x + cols

        return x


def _context_aggregating_cnn(x, filters, name):
    """A stack of dilated convs that rapidly expands the receptive field,
    approximating MetNet's large-context conv tower."""
    dilations = [1, 2, 4, 8, 16, 1]
    for i, d in enumerate(dilations):
        x = layers.Conv2D(filters, 3, padding="same", dilation_rate=d, name=f"{name}_dconv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    return x


def MetNet(
    input_shape=(7, 256, 256, 1),  # (T_in, H, W, C) - e.g. satellite + radar stack
    lead_times=12,
    base_filters=96,
    attn_dim=256,
    attn_layers=4,
    num_heads=8,
    downsample_factor=4,
    num_bins=1,  # e.g. precipitation-rate bins for a categorical head; 1 => regression
    name="metnet",
):
    T_in, H, W, C_in = input_shape
    inputs = keras.Input(shape=input_shape, name="frames")
    lead_time_input = keras.Input(shape=(), dtype="int32", name="lead_time_idx")

    # --- per-frame conv stem + downsample (patchify to a manageable grid) ---
    # Every layer wrapped in `TimeDistributed` below gets an explicit inner
    # `name=` - otherwise Keras auto-assigns globally-incrementing names
    # ("conv2d", "batch_normalization", ...) that depend on how many other
    # unnamed layers of that type exist earlier in the same process, which
    # makes weight-porting by name unreproducible.
    # padding="same" would pad asymmetrically (or by the wrong total
    # amount) for a strided conv like this one - explicit symmetric
    # padding (matching PyTorch's conventional `padding=kernel//2`) is
    # needed for faithful weight porting from a real checkpoint (see
    # `ConvBNAct` in utils/layers.py for the fuller explanation).
    x = layers.TimeDistributed(layers.ZeroPadding2D(1), name="stem_pad")(inputs)
    x = layers.TimeDistributed(
        layers.Conv2D(base_filters, 3, strides=downsample_factor, padding="valid", name="conv"),
        name="stem",
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization(name="bn"), name="stem_bn")(x)
    x = layers.TimeDistributed(layers.Activation("relu", name="act"), name="stem_act")(x)

    # --- temporal encoding: collapse T_in frames with a ConvLSTM ---
    x = layers.ConvLSTM2D(base_filters, 3, padding="same", return_sequences=False,
                           name="temporal_encoder")(x)

    # --- large-context conv tower (dilated convs) ---
    x = _context_aggregating_cnn(x, base_filters * 2, name="context_tower")

    # --- lead-time conditioning: embed the target lead time and broadcast-add ---
    lead_embed = layers.Embedding(lead_times, base_filters * 2, name="lead_time_embed")(lead_time_input)
    lead_embed = layers.Reshape((1, 1, base_filters * 2), name="lead_time_reshape")(lead_embed)
    x = layers.Add(name="add_lead_time")([x, lead_embed])

    # --- axial self-attention aggregator ---
    x = layers.Conv2D(attn_dim, 1, name="attn_proj_in")(x)
    for i in range(attn_layers):
        x = AxialAttention2D(attn_dim, num_heads, name=f"axial_attn{i}")(x)

    # --- upsample back to input resolution and predict ---
    # kernel_size == strides (2, not the more common 3x3) is deliberate:
    # transposed-conv padding alignment between "same"-style frameworks and
    # an explicit-padding PyTorch reference is not standardized the way
    # regular-conv padding is (see ConvBNAct's docstring for the regular
    # case), but kernel==stride with zero padding is unambiguous on both
    # sides - the same convention already proven exact for UNet's
    # transposed convs when porting a real checkpoint.
    for i in range(int(downsample_factor).bit_length() - 1):
        x = layers.Conv2DTranspose(attn_dim // 2, 2, strides=2, padding="same",
                                    name=f"upsample{i}")(x)
        x = layers.BatchNormalization(name=f"upsample_bn{i}")(x)
        x = layers.Activation("relu", name=f"upsample_relu{i}")(x)

    activation = "softmax" if num_bins > 1 else "sigmoid"
    outputs = layers.Conv2D(num_bins, 1, activation=activation, name="precip_head")(x)

    return keras.Model([inputs, lead_time_input], outputs, name=name)
