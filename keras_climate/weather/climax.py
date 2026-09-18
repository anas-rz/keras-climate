"""
keras_climate.weather.climax
--------------------------------
ClimaX (Nguyen et al. 2023, Microsoft): a ViT-based foundation model for
weather/climate, trained across heterogeneous datasets (CMIP6, ERA5) with
varying variable sets and resolutions. Each input variable/channel gets
its own patch embedding (a separate `Conv2D` per variable, not one shared
conv), a learned per-variable embedding distinguishes them, then a
single-query cross-attention ("variable aggregation") pools the per-
variable token sequences into one unified sequence before a standard ViT
encoder; a lead-time embedding (a plain scalar -> vector `Dense`) lets one
model serve multiple forecast horizons.

Real pretrained checkpoint: Microsoft's official release, hosted directly
on HuggingFace (`microsoft/ClimaX`, `1.40625deg.ckpt`) - see
`weights/pretrained.py`'s `climax_1_40625deg` loader.
"""

import keras
from keras import layers, ops
from keras_climate.utils.layers import TransformerEncoderBlock, unpatchify_2d


class MultiVariablePatchEmbed(layers.Layer):
    """Per-variable patch embedding: each of `num_vars` input channels
    gets its OWN separate `Conv2D` (matching the official ClimaX's
    `token_embeds` - a `ModuleList` of per-variable `PatchEmbed`
    instances, not a single shared conv), tokenized and stacked along a
    new variable axis. `(B, H, W, num_vars)` -> `(B, num_vars, L,
    embed_dim)`."""

    def __init__(self, num_vars, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_vars = num_vars
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.token_embeds = [
            layers.Conv2D(self.embed_dim, self.patch_size, strides=self.patch_size,
                           name=f"token_embeds{v}")
            for v in range(self.num_vars)
        ]
        super().build(input_shape)

    def call(self, x):
        embeds = []
        for v in range(self.num_vars):
            t = self.token_embeds[v](x[..., v:v + 1])  # (B, gh, gw, D)
            B, gh, gw = ops.shape(t)[0], t.shape[1], t.shape[2]
            embeds.append(ops.reshape(t, (B, gh * gw, self.embed_dim)))
        return ops.stack(embeds, axis=1)  # (B, V, L, D)


class VariableEmbedding(layers.Layer):
    """A learned per-variable embedding vector, added to every patch
    token of its variable (broadcast over the patch/L axis) - matches the
    official `var_embed` parameter."""

    def __init__(self, num_vars, dim, **kwargs):
        super().__init__(**kwargs)
        self.num_vars = num_vars
        self.dim = dim

    def build(self, input_shape):
        self.var_embed = self.add_weight(shape=(1, self.num_vars, self.dim),
                                          initializer="zeros", trainable=True, name="var_embed")
        super().build(input_shape)

    def call(self, x):
        # x: (B, V, L, D); var_embed broadcasts over L.
        return x + self.var_embed[:, :, None, :]


class VariableAggregation(layers.Layer):
    """Cross-attention with a single learnable query per spatial
    location, pooling over the variable axis - matches the official
    `aggregate_variables`'s `nn.MultiheadAttention(embed_dim, num_heads,
    batch_first=True)` called as `var_agg(var_query, x, x)`, including its
    exact combined-QKV weight layout (`in_proj_weight`/`in_proj_bias` -
    one matrix split into three `embed_dim`-sized chunks for Q/K/V - plus
    `out_proj`). `(B, V, L, D) -> (B, L, D)`."""

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

    def build(self, input_shape):
        D = self.embed_dim
        self.var_query = self.add_weight(shape=(1, 1, D), initializer="zeros",
                                          trainable=True, name="var_query")
        self.in_proj_weight = self.add_weight(shape=(3 * D, D), initializer="glorot_uniform",
                                               name="in_proj_weight")
        self.in_proj_bias = self.add_weight(shape=(3 * D,), initializer="zeros",
                                             name="in_proj_bias")
        self.out_proj = layers.Dense(D, name="out_proj")
        super().build(input_shape)

    def call(self, x):
        D = self.embed_dim
        B, V, L = ops.shape(x)[0], x.shape[1], x.shape[2]
        x_t = ops.transpose(x, (0, 2, 1, 3))  # (B, L, V, D)
        x_flat = ops.reshape(x_t, (B * L, V, D))

        wq, wk, wv = self.in_proj_weight[:D], self.in_proj_weight[D:2 * D], self.in_proj_weight[2 * D:]
        bq, bk, bv = self.in_proj_bias[:D], self.in_proj_bias[D:2 * D], self.in_proj_bias[2 * D:]

        q = ops.matmul(self.var_query, ops.transpose(wq)) + bq  # (1, 1, D)
        q = ops.broadcast_to(q, (B * L, 1, D))
        k = ops.matmul(x_flat, ops.transpose(wk)) + bk  # (B*L, V, D)
        v = ops.matmul(x_flat, ops.transpose(wv)) + bv

        q = ops.transpose(ops.reshape(q, (B * L, 1, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(k, (B * L, V, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(v, (B * L, V, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        attn = ops.softmax(ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale, axis=-1)
        out = ops.matmul(attn, v)  # (B*L, heads, 1, head_dim)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B * L, 1, D))
        out = self.out_proj(out[:, 0, :])  # (B*L, D)
        return ops.reshape(out, (B, L, D))


class LearnedPositionEmbedding1D(layers.Layer):
    """Learned (not fixed) absolute positional embedding, matches the
    official `pos_embed` parameter (fixed-sincos-*initialized* upstream,
    but trainable/loaded from checkpoint, not recomputed)."""

    def __init__(self, num_patches, dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.dim = dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(shape=(1, self.num_patches, self.dim),
                                          initializer="zeros", trainable=True, name="pos_embed")
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_embed


class AddLeadTime(layers.Layer):
    """Broadcasts the lead-time embedding `(B, D)` over the patch/L axis
    and adds it to the token sequence `(B, L, D)`."""

    def call(self, inputs):
        x, lead_embed = inputs
        return x + lead_embed[:, None, :]


def ClimaX(img_size=(128, 256), patch_size=4, num_vars=48, embed_dim=1024, depth=8,
           decoder_depth=2, num_heads=16, mlp_ratio=4.0, name="climax"):
    """Inputs:
        fields: (B, H, W, num_vars) - one timestep's stacked variables.
        lead_time: (B, 1) - forecast lead time (in whatever units the
            checkpoint was trained with, typically hours/24).
    Output: (B, H, W, num_vars) - the predicted state at `lead_time`.
    """
    H, W = img_size
    grid_h, grid_w = H // patch_size, W // patch_size
    num_patches = grid_h * grid_w

    field_in = keras.Input(shape=(H, W, num_vars), name="fields")
    lead_time_in = keras.Input(shape=(1,), name="lead_time")

    x = MultiVariablePatchEmbed(num_vars, patch_size, embed_dim, name="token_embeds")(field_in)
    x = VariableEmbedding(num_vars, embed_dim, name="var_embed_layer")(x)
    x = VariableAggregation(embed_dim, num_heads, name="var_agg")(x)  # (B, L, D)
    x = LearnedPositionEmbedding1D(num_patches, embed_dim, name="pos_embed_layer")(x)

    lead_embed = layers.Dense(embed_dim, name="lead_time_embed")(lead_time_in)  # (B, D)
    x = AddLeadTime(name="add_lead_time")([x, lead_embed])

    for i in range(depth):
        x = TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, name=f"blocks{i}")(x)
    x = layers.LayerNormalization(name="norm")(x)

    for i in range(decoder_depth):
        x = layers.Dense(embed_dim, activation="gelu", name=f"head{2 * i}")(x)
    out = layers.Dense(num_vars * patch_size ** 2, name=f"head{2 * decoder_depth}")(x)

    out = layers.Reshape((grid_h, grid_w, patch_size * patch_size * num_vars), name="to_grid")(out)
    out = unpatchify_2d(out, grid_h, grid_w, patch_size, num_vars)

    return keras.Model([field_in, lead_time_in], out, name=name)
