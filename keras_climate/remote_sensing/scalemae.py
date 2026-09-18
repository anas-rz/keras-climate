"""
keras_climate.remote_sensing.scalemae
-----------------------------------------
Scale-MAE (Reed et al. 2023): a Masked Autoencoder ViT whose positional
embedding is a function of each image's ground-sample-distance (GSD, in
meters/pixel) rather than a fixed learned/sin-cos table. This makes the
same pretrained encoder usable across satellite/aerial imagery captured at
different physical resolutions - the pixel grid stays the same size but
the *scale* it represents is baked into the positional encoding.

Only the encoder (`ScaleMAEEncoder`) is implemented here, for downstream
feature extraction - the official pretraining pipeline also includes a
Laplacian-pyramid FPN decoder used solely to compute the MAE reconstruction
loss, which is out of scope for a feature-extraction library (mirrors this
repo's `SatMAEEncoder`-only usage pattern).
"""

import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import PatchEmbed2D, TransformerEncoderBlock


class GSDPositionalEmbedding(layers.Layer):
    """2D sin-cos positional embedding whose spatial grid coordinates are
    scaled by each sample's ground-sample-distance (meters/pixel) before
    the sin/cos projection - so the same `grid_size x grid_size` token grid
    represents a different physical extent depending on input resolution.
    Same sin/cos math as `sincos_position_embedding_2d`, just parameterized
    by a per-sample `res` instead of a fixed grid index."""

    def __init__(self, grid_size, dim, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.dim = dim
        grid_h, grid_w = np.meshgrid(
            np.arange(grid_size, dtype=np.float32),
            np.arange(grid_size, dtype=np.float32),
            indexing="ij",
        )
        self._grid_h_np = grid_h.reshape(-1)  # (N,)
        self._grid_w_np = grid_w.reshape(-1)
        self._omega_np = 1.0 / (10000 ** (np.arange(dim // 4, dtype=np.float32) / (dim / 4.0)))

    def build(self, input_shape):
        self.grid_h = self.add_weight(shape=self._grid_h_np.shape, initializer=keras.initializers.Constant(
            self._grid_h_np), trainable=False, name="grid_h")
        self.grid_w = self.add_weight(shape=self._grid_w_np.shape, initializer=keras.initializers.Constant(
            self._grid_w_np), trainable=False, name="grid_w")
        self.omega = self.add_weight(shape=self._omega_np.shape, initializer=keras.initializers.Constant(
            self._omega_np), trainable=False, name="omega")
        super().build(input_shape)

    def call(self, res):
        # res: (B,) ground-sample-distance, meters/pixel
        gh = self.grid_h[None, :] * res[:, None]  # (B, N)
        gw = self.grid_w[None, :] * res[:, None]
        out_h = gh[..., None] * self.omega[None, None, :]  # (B, N, dim//4)
        out_w = gw[..., None] * self.omega[None, None, :]
        emb_h = ops.concatenate([ops.sin(out_h), ops.cos(out_h)], axis=-1)  # (B, N, dim//2)
        emb_w = ops.concatenate([ops.sin(out_w), ops.cos(out_w)], axis=-1)
        pos = ops.concatenate([emb_h, emb_w], axis=-1)  # (B, N, dim)
        cls_pos = ops.zeros((ops.shape(res)[0], 1, self.dim), dtype=pos.dtype)
        return ops.concatenate([cls_pos, pos], axis=1)


class ScaleMAEEncoder(keras.Model):
    """ViT-L/16 encoder with GSD-aware positional embedding. Call with
    `(image, res)` where `res` is a (B,) tensor of meters/pixel; pass a
    constant array (e.g. `np.full((B,), 1.0)`) if GSD is unknown/fixed -
    ScaleMAE's official fMoW-RGB checkpoint was trained with `res` in
    roughly [0.3, 5.0] meters/pixel."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024,
                 depth=24, num_heads=16, mlp_ratio=4.0, name="scalemae_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        grid_size = img_size // patch_size

        self.patch_embed = PatchEmbed2D(patch_size, embed_dim, norm=False, name="patch_embed")
        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer="zeros",
                                          trainable=True, name="cls_token")
        self.pos_embed = GSDPositionalEmbedding(grid_size, embed_dim, name="pos_embed")

        self.blocks = [
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, name=f"block{i}")
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, inputs, training=False):
        x, res = inputs
        tokens, H, W = self.patch_embed(x)
        pos = self.pos_embed(res)  # (B, 1+N, D)

        B = ops.shape(tokens)[0]
        cls = ops.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        tokens = ops.concatenate([cls, tokens], axis=1)
        tokens = tokens + pos

        for blk in self.blocks:
            tokens = blk(tokens, training=training)
        return self.norm(tokens)


def ScaleMAE(img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
             num_heads=16, mlp_ratio=4.0, name="scalemae"):
    """Feature-extractor model: `[image, res] -> tokens (B, 1+N, embed_dim)`."""
    image_in = keras.Input(shape=(img_size, img_size, in_chans), name="image")
    res_in = keras.Input(shape=(), name="res")
    encoder = ScaleMAEEncoder(img_size, patch_size, in_chans, embed_dim, depth,
                               num_heads, mlp_ratio, name=f"{name}_encoder")
    tokens = encoder([image_in, res_in])
    return keras.Model([image_in, res_in], tokens, name=name)
