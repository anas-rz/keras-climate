import keras
from keras import layers, ops
from keras_climate.utils.layers import MLP, rfft2_hw, irfft2_hw, unpatchify_2d


class AFNO2D(layers.Layer):

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
                 hidden_size_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold

    def build(self, input_shape):
        scale = 0.02
        bs, hf, nb = self.block_size, self.hidden_size_factor, self.num_blocks
        init = keras.initializers.RandomNormal(stddev=scale)
        self.w1 = self.add_weight(shape=(2, nb, bs, bs * hf), initializer=init, name="w1")
        self.b1 = self.add_weight(shape=(2, nb, bs * hf), initializer=init, name="b1")
        self.w2 = self.add_weight(shape=(2, nb, bs * hf, bs), initializer=init, name="w2")
        self.b2 = self.add_weight(shape=(2, nb, bs), initializer=init, name="b2")
        super().build(input_shape)

    def call(self, x):
        bias = x
        H, W = x.shape[1], x.shape[2]
        re, im = rfft2_hw(x)

        shape = ops.shape(re)
        B, Wf = shape[0], shape[2]
        re = ops.reshape(re, (B, H, Wf, self.num_blocks, self.block_size))
        im = ops.reshape(im, (B, H, Wf, self.num_blocks, self.block_size))

        o1_re = ops.relu(
            ops.einsum("bhwnc,ncd->bhwnd", re, self.w1[0])
            - ops.einsum("bhwnc,ncd->bhwnd", im, self.w1[1])
            + self.b1[0]
        )
        o1_im = ops.relu(
            ops.einsum("bhwnc,ncd->bhwnd", im, self.w1[0])
            + ops.einsum("bhwnc,ncd->bhwnd", re, self.w1[1])
            + self.b1[1]
        )

        o2_re = (
            ops.einsum("bhwnd,nde->bhwne", o1_re, self.w2[0])
            - ops.einsum("bhwnd,nde->bhwne", o1_im, self.w2[1])
            + self.b2[0]
        )
        o2_im = (
            ops.einsum("bhwnd,nde->bhwne", o1_im, self.w2[0])
            + ops.einsum("bhwnd,nde->bhwne", o1_re, self.w2[1])
            + self.b2[1]
        )

        o2_re = ops.reshape(o2_re, (B, H, Wf, self.hidden_size))
        o2_im = ops.reshape(o2_im, (B, H, Wf, self.hidden_size))

        lam = self.sparsity_threshold
        o2_re = ops.sign(o2_re) * ops.relu(ops.abs(o2_re) - lam)
        o2_im = ops.sign(o2_im) * ops.relu(ops.abs(o2_im) - lam)

        out = irfft2_hw(o2_re, o2_im, H, W)
        return out + bias


class AFNOBlock(layers.Layer):

    def __init__(self, hidden_size, num_blocks=8, mlp_ratio=4.0, sparsity_threshold=0.01,
                 hidden_size_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.filter = AFNO2D(hidden_size, num_blocks, sparsity_threshold, hidden_size_factor,
                              name="filter")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.mlp = MLP(int(hidden_size * mlp_ratio), hidden_size, name="mlp")

    def call(self, x, training=False):
        x = x + self.filter(self.norm1(x))
        x = x + self.mlp(self.norm2(x), training=training)
        return x


def AFNOOperator(input_shape=(64, 64, 3), out_channels=3, patch_size=8, embed_dim=256,
                  depth=8, num_blocks=8, mlp_ratio=4.0, name="afno_operator"):
    grid = input_shape[0] // patch_size
    inputs = keras.Input(shape=input_shape, name="field")
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size,
                       name="patch_embed_proj")(inputs)

    for i in range(depth):
        x = AFNOBlock(embed_dim, num_blocks, mlp_ratio, name=f"block{i}")(x)

    x = layers.Dense(patch_size * patch_size * out_channels, name="head")(x)
    x = unpatchify_2d(x, grid, grid, patch_size, out_channels)
    return keras.Model(inputs, x, name=name)
