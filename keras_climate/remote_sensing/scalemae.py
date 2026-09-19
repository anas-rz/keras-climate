import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import PatchEmbed2D, TransformerEncoderBlock


class GSDPositionalEmbedding(layers.Layer):

    def __init__(self, grid_size, dim, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.dim = dim
        grid_h, grid_w = np.meshgrid(
            np.arange(grid_size, dtype=np.float32),
            np.arange(grid_size, dtype=np.float32),
            indexing="ij",
        )
        self._grid_h_np = grid_h.reshape(-1)
        self._grid_w_np = grid_w.reshape(-1)
        self._omega_np = 1.0 / (
            10000 ** (np.arange(dim // 4, dtype=np.float32) / (dim / 4.0))
        )

    def build(self, input_shape):
        self.grid_h = self.add_weight(
            shape=self._grid_h_np.shape,
            initializer=keras.initializers.Constant(self._grid_h_np),
            trainable=False,
            name="grid_h",
        )
        self.grid_w = self.add_weight(
            shape=self._grid_w_np.shape,
            initializer=keras.initializers.Constant(self._grid_w_np),
            trainable=False,
            name="grid_w",
        )
        self.omega = self.add_weight(
            shape=self._omega_np.shape,
            initializer=keras.initializers.Constant(self._omega_np),
            trainable=False,
            name="omega",
        )
        super().build(input_shape)

    def call(self, res):
        gh = self.grid_h[None, :] * res[:, None]
        gw = self.grid_w[None, :] * res[:, None]
        out_h = gh[..., None] * self.omega[None, None, :]
        out_w = gw[..., None] * self.omega[None, None, :]
        emb_h = ops.concatenate([ops.sin(out_h), ops.cos(out_h)], axis=-1)
        emb_w = ops.concatenate([ops.sin(out_w), ops.cos(out_w)], axis=-1)
        pos = ops.concatenate([emb_h, emb_w], axis=-1)
        cls_pos = ops.zeros((ops.shape(res)[0], 1, self.dim), dtype=pos.dtype)
        return ops.concatenate([cls_pos, pos], axis=1)


class ScaleMAEEncoder(keras.Model):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        name="scalemae_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        grid_size = img_size // patch_size

        self.patch_embed = PatchEmbed2D(
            patch_size, embed_dim, norm=False, name="patch_embed"
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token",
        )
        self.pos_embed = GSDPositionalEmbedding(grid_size, embed_dim, name="pos_embed")

        self.blocks = [
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, name=f"block{i}")
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, inputs, training=False):
        x, res = inputs
        tokens, H, W = self.patch_embed(x)
        pos = self.pos_embed(res)

        B = ops.shape(tokens)[0]
        cls = ops.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        tokens = ops.concatenate([cls, tokens], axis=1)
        tokens = tokens + pos

        for blk in self.blocks:
            tokens = blk(tokens, training=training)
        return self.norm(tokens)


def ScaleMAE(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4.0,
    name="scalemae",
):
    image_in = keras.Input(shape=(img_size, img_size, in_chans), name="image")
    res_in = keras.Input(shape=(), name="res")
    encoder = ScaleMAEEncoder(
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        name=f"{name}_encoder",
    )
    tokens = encoder([image_in, res_in])
    return keras.Model([image_in, res_in], tokens, name=name)
