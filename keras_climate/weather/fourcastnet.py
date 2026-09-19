import keras
from keras import layers
from keras_climate.operators.afno import AFNOBlock
from keras_climate.utils.layers import unpatchify_2d


class LearnedPositionEmbedding2D(layers.Layer):

    def __init__(self, grid_h, grid_w, dim, **kwargs):
        super().__init__(**kwargs)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.dim = dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            shape=(1, self.grid_h, self.grid_w, self.dim),
            initializer="zeros",
            trainable=True,
            name="pos_embed",
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_embed


def FourCastNet(
    img_size=(720, 1440),
    patch_size=8,
    in_chans=20,
    out_chans=20,
    embed_dim=768,
    depth=12,
    num_blocks=8,
    mlp_ratio=4.0,
    name="fourcastnet",
):
    H, W = img_size
    grid_h, grid_w = H // patch_size, W // patch_size

    inputs = keras.Input(shape=(H, W, in_chans), name="state")
    x = layers.Conv2D(
        embed_dim, kernel_size=patch_size, strides=patch_size, name="patch_embed_proj"
    )(inputs)
    x = LearnedPositionEmbedding2D(grid_h, grid_w, embed_dim, name="pos_embed_layer")(x)

    for i in range(depth):
        x = AFNOBlock(embed_dim, num_blocks, mlp_ratio, name=f"block{i}")(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="norm")(x)
    x = layers.Dense(patch_size * patch_size * out_chans, use_bias=False, name="head")(
        x
    )
    x = unpatchify_2d(x, grid_h, grid_w, patch_size, out_chans)

    return keras.Model(inputs, x, name=name)
