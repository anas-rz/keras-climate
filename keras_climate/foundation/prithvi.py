"""
keras_climate.foundation.prithvi
------------------------------------
Prithvi (IBM/NASA, 2023): a ViT Masked Autoencoder pretrained on NASA
Harmonized Landsat-Sentinel (HLS) imagery, using 3D (tubelet) patch
embedding over a short temporal stack (typically 3 timesteps) of 6-band
multispectral imagery. Downstream tasks (e.g. burn-scar or flood mapping)
attach a segmentation/classification head to `PrithviEncoder`.
"""

import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import PatchEmbed3D, TransformerEncoderBlock, sincos_position_embedding


PRITHVI_CONFIGS = {
    "prithvi_100m": dict(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512,
                          decoder_depth=8, decoder_num_heads=16),
    "prithvi_300m": dict(embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512,
                          decoder_depth=8, decoder_num_heads=16),
}


class PrithviEncoder(keras.Model):
    """3D-patch ViT encoder over a (B, T, H, W, C) HLS image stack."""

    def __init__(self, img_size=224, patch_size=16, num_frames=3, tubelet_size=1,
                 in_chans=6, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 name="prithvi_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = img_size // patch_size
        self.t_grid = num_frames // tubelet_size

        self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, embed_dim, name="patch_embed")
        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer="zeros",
                                          trainable=True, name="cls_token")

        num_patches = self.t_grid * self.grid_size * self.grid_size
        # Factorized spatiotemporal sin-cos position embedding: spatial (2D) x temporal (1D)
        spatial_pos = sincos_position_embedding(self.grid_size * self.grid_size, embed_dim // 4 * 4)
        temporal_pos = sincos_position_embedding(self.t_grid, embed_dim)
        pos = np.zeros((num_patches, embed_dim), dtype=np.float32)
        for t in range(self.t_grid):
            for s in range(self.grid_size * self.grid_size):
                pos[t * self.grid_size * self.grid_size + s] = temporal_pos[t] + np.pad(
                    spatial_pos[s], (0, embed_dim - spatial_pos.shape[1]))
        pos = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos], axis=0)
        self.pos_embed = self.add_weight(shape=pos.shape, initializer=keras.initializers.Constant(pos),
                                          trainable=False, name="pos_embed")

        self.blocks = [TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, name=f"block{i}")
                        for i in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, x, training=False, return_all_tokens=True):
        tokens, T, H, W = self.patch_embed(x)
        tokens = tokens + self.pos_embed[None, 1:, :]

        B = ops.shape(tokens)[0]
        cls = ops.broadcast_to(self.cls_token + self.pos_embed[None, :1, :], (B, 1, self.embed_dim))
        tokens = ops.concatenate([cls, tokens], axis=1)

        for blk in self.blocks:
            tokens = blk(tokens, training=training)
        tokens = self.norm(tokens)

        if return_all_tokens:
            return tokens
        return tokens[:, 0]  # cls token only


def PrithviClassifier(variant="prithvi_100m", img_size=224, patch_size=16, num_frames=3,
                       in_chans=6, num_classes=10, name="prithvi_classifier"):
    """Prithvi encoder + linear probe / fine-tune head for scene classification."""
    cfg = PRITHVI_CONFIGS[variant]
    inputs = keras.Input(shape=(num_frames, img_size, img_size, in_chans), name="hls_stack")
    encoder = PrithviEncoder(img_size, patch_size, num_frames, 1, in_chans,
                              cfg["embed_dim"], cfg["depth"], cfg["num_heads"], name="encoder")
    cls_token = encoder(inputs, return_all_tokens=False)
    outputs = layers.Dense(num_classes, name="head")(cls_token)
    return keras.Model(inputs, outputs, name=name)


def PrithviSegmenter(variant="prithvi_100m", img_size=224, patch_size=16, num_frames=3,
                      in_chans=6, num_classes=2, name="prithvi_segmenter"):
    """Prithvi encoder + a simple conv decode head, for dense prediction
    tasks (burn scar / flood / crop-type mapping)."""
    cfg = PRITHVI_CONFIGS[variant]
    inputs = keras.Input(shape=(num_frames, img_size, img_size, in_chans), name="hls_stack")
    encoder = PrithviEncoder(img_size, patch_size, num_frames, 1, in_chans,
                              cfg["embed_dim"], cfg["depth"], cfg["num_heads"], name="encoder")
    tokens = encoder(inputs, return_all_tokens=True)
    patch_tokens = layers.Lambda(lambda t: t[:, 1:, :], name="drop_cls")(tokens)

    grid = img_size // patch_size
    t_grid = num_frames
    x = layers.Reshape((t_grid, grid, grid, cfg["embed_dim"]), name="to_grid")(patch_tokens)
    x = layers.Lambda(lambda t: ops.mean(t, axis=1), name="temporal_pool")(x)  # fuse time

    for i, f in enumerate([256, 128, 64]):
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same", name=f"up{i}")(x)
        x = layers.BatchNormalization(name=f"up_bn{i}")(x)
        x = layers.Activation("relu", name=f"up_relu{i}")(x)

    x = layers.Lambda(lambda t: ops.image.resize(t, (img_size, img_size), interpolation="bilinear"),
                       name="resize_to_input")(x)
    outputs = layers.Conv2D(num_classes, 1, name="seg_head")(x)
    return keras.Model(inputs, outputs, name=name)
