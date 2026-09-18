"""
keras_climate.foundation.prithvi
------------------------------------
Prithvi (IBM/NASA, 2023): a ViT Masked Autoencoder pretrained on NASA
Harmonized Landsat-Sentinel (HLS) imagery, using 3D (tubelet) patch
embedding over a short temporal stack (typically 3 timesteps) of 6-band
multispectral imagery. Downstream tasks (e.g. burn-scar or flood mapping)
attach a segmentation/classification head to `PrithviEncoder`.

Prithvi-EO-2.0 (the 2024 successor, `prithvi_eo_v2_300m`/`_600m` below)
reuses this exact same `PrithviEncoder` architecture unchanged - same
patch embedding, same factorized 3D sin-cos positional embedding, same
ViT block - just at different embed_dim/depth/num_heads/patch_size/
num_frames, so no new model code is needed for its base (non-"-TL")
checkpoint variants (see `weights/pretrained.py`'s `prithvi_eo_v2_300m`
loader). The "-TL" (Temporal+Location) variants additionally add a
`TemporalEncoder`/`LocationEncoder` (each just one small learned/fixed
per-channel scale tensor encoding acquisition timestamp/lat-lon) that
this module does not yet implement.
"""

import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import PatchEmbed3D, TransformerEncoderBlock


PRITHVI_CONFIGS = {
    "prithvi_100m": dict(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512,
                          decoder_depth=8, decoder_num_heads=16),
    "prithvi_300m": dict(embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512,
                          decoder_depth=8, decoder_num_heads=16),
    # Prithvi-EO-2.0 (see module docstring) - same architecture, larger.
    "prithvi_eo_v2_300m": dict(embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512,
                                decoder_depth=8, decoder_num_heads=16, patch_size=16),
    "prithvi_eo_v2_600m": dict(embed_dim=1280, depth=32, num_heads=16, decoder_embed_dim=512,
                                decoder_depth=8, decoder_num_heads=16, patch_size=14),
}


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Matches the official Prithvi/MAE-ST `get_1d_sincos_pos_embed_from_grid`
    exactly (needed bit-for-bit, since `pos_embed` ships as a checkpoint
    buffer computed by this same formula)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1).astype(np.float64)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


def get_3d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """Factorized 3D (t, h, w) sin-cos position embedding: separate 1D
    sin-cos embeddings per axis, split `embed_dim` into w:h:t = 6:6:4
    sixteenths and *concatenated* (not summed) along the feature axis -
    matches the official Prithvi `get_3d_sincos_pos_embed` exactly (this is
    a frozen/non-trainable buffer in the source checkpoint, so getting this
    formula bit-exact is what lets the encoder's pos_embed be ported - or
    equivalently, recomputed identically - without any learned weights of
    its own)."""
    assert embed_dim % 16 == 0
    t_size, h_size, w_size = grid_size
    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = _get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = _get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = _get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate([w_pos_embed, h_pos_embed, t_pos_embed], axis=1)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0)
    return pos_embed


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

        pos = get_3d_sincos_pos_embed(embed_dim, (self.t_grid, self.grid_size, self.grid_size),
                                       add_cls_token=True)
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
