"""
keras_climate.weather.fourcastnet
-------------------------------------
FourCastNet (Pathak et al. 2022, NVIDIA): a ViT-style global weather
forecaster using AFNO (see `keras_climate.operators.afno`) as its token-
mixer instead of self-attention, giving O(N log N) spatial mixing suited
to the large token grids a 0.25-degree-resolution ERA5 forecast needs
(720x1440 pixels, still a large token grid after patchifying). Trained to
predict the next 6-hour ERA5 atmospheric state from the current one
(autoregressive rollout at inference for multi-step forecasts); the same
architecture also has a released "precip" variant fine-tuned to predict
total precipitation instead.

Real pretrained checkpoint: NVIDIA's official release, mirrored non-
interactively at NERSC (`https://portal.nersc.gov/project/m4134/
FCN_weights_v0/{backbone,precip}.ckpt`) - see `weights/pretrained.py`'s
`fourcastnet_backbone` loader. Only `hard_thresholding_fraction=1.0` is
supported (matches the released checkpoint's own config - see
`keras_climate.operators.afno`'s module docstring for why).
"""

import keras
from keras import layers
from keras_climate.operators.afno import AFNOBlock
from keras_climate.utils.layers import unpatchify_2d


class LearnedPositionEmbedding2D(layers.Layer):
    """Learned (not fixed sin-cos) absolute positional embedding added to
    the patch-token grid right after patch embedding - matches the
    official FourCastNet's `pos_embed` (stored flattened as `(1, N, C)` in
    the checkpoint; reshaped to `(1, grid_h, grid_w, C)` when porting, to
    match this module's `(B, H, W, C)` token-grid layout throughout - see
    `weights/mappings/fourcastnet_mapping.py`)."""

    def __init__(self, grid_h, grid_w, dim, **kwargs):
        super().__init__(**kwargs)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.dim = dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(shape=(1, self.grid_h, self.grid_w, self.dim),
                                          initializer="zeros", trainable=True, name="pos_embed")
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_embed


def FourCastNet(img_size=(720, 1440), patch_size=8, in_chans=20, out_chans=20,
                 embed_dim=768, depth=12, num_blocks=8, mlp_ratio=4.0, name="fourcastnet"):
    """Inputs: `state` of shape `(B, H, W, in_chans)` - one ERA5 timestep's
    stacked atmospheric/surface variables. Output: the predicted next
    timestep, same shape (with `out_chans` channels)."""
    H, W = img_size
    grid_h, grid_w = H // patch_size, W // patch_size

    inputs = keras.Input(shape=(H, W, in_chans), name="state")
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size,
                       name="patch_embed_proj")(inputs)
    x = LearnedPositionEmbedding2D(grid_h, grid_w, embed_dim, name="pos_embed_layer")(x)

    for i in range(depth):
        x = AFNOBlock(embed_dim, num_blocks, mlp_ratio, name=f"block{i}")(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="norm")(x)
    x = layers.Dense(patch_size * patch_size * out_chans, use_bias=False, name="head")(x)
    x = unpatchify_2d(x, grid_h, grid_w, patch_size, out_chans)

    return keras.Model(inputs, x, name=name)
