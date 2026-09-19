import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import rfft2_hw, irfft2_hw


class CoordinateGrid(layers.Layer):

    def __init__(self, H, W, **kwargs):
        super().__init__(**kwargs)
        gx, gy = np.meshgrid(np.linspace(0, 1, H, dtype="float32"),
                              np.linspace(0, 1, W, dtype="float32"), indexing="ij")
        self._grid_np = np.stack([gx, gy], axis=-1)

    def build(self, input_shape):
        self.grid = self.add_weight(shape=self._grid_np.shape,
                                     initializer=keras.initializers.Constant(self._grid_np),
                                     trainable=False, name="grid")
        super().build(input_shape)

    def call(self, x):
        B = ops.shape(x)[0]
        grid = ops.broadcast_to(self.grid[None], (B,) + tuple(self.grid.shape))
        return ops.concatenate([x, grid], axis=-1)


class SpectralConv2D(layers.Layer):

    def __init__(self, in_channels, out_channels, modes1, modes2, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def build(self, input_shape):
        scale = 1.0 / (self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, self.modes1, self.modes2)
        init = keras.initializers.RandomUniform(minval=0.0, maxval=scale)
        self.weights1_re = self.add_weight(shape=shape, initializer=init, name="weights1_re")
        self.weights1_im = self.add_weight(shape=shape, initializer=init, name="weights1_im")
        self.weights2_re = self.add_weight(shape=shape, initializer=init, name="weights2_re")
        self.weights2_im = self.add_weight(shape=shape, initializer=init, name="weights2_im")
        super().build(input_shape)

    @staticmethod
    def _compl_mul2d(in_re, in_im, w_re, w_im):
        out_re = ops.einsum("bixy,ioxy->boxy", in_re, w_re) - ops.einsum("bixy,ioxy->boxy", in_im, w_im)
        out_im = ops.einsum("bixy,ioxy->boxy", in_re, w_im) + ops.einsum("bixy,ioxy->boxy", in_im, w_re)
        return out_re, out_im

    def call(self, x):
        H, W = x.shape[1], x.shape[2]
        re, im = rfft2_hw(x)
        re = ops.transpose(re, (0, 3, 1, 2))
        im = ops.transpose(im, (0, 3, 1, 2))
        Wf = re.shape[-1]

        top_re, top_im = re[:, :, :self.modes1, :self.modes2], im[:, :, :self.modes1, :self.modes2]
        bot_re, bot_im = re[:, :, -self.modes1:, :self.modes2], im[:, :, -self.modes1:, :self.modes2]

        out_top_re, out_top_im = self._compl_mul2d(top_re, top_im, self.weights1_re, self.weights1_im)
        out_bot_re, out_bot_im = self._compl_mul2d(bot_re, bot_im, self.weights2_re, self.weights2_im)

        B = ops.shape(re)[0]
        mid_rows = H - 2 * self.modes1
        pad_cols = Wf - self.modes2
        pad_spec = [[0, 0], [0, 0], [0, 0], [0, pad_cols]]
        mid = ops.zeros((B, self.out_channels, mid_rows, Wf))

        out_re_full = ops.concatenate([ops.pad(out_top_re, pad_spec), mid, ops.pad(out_bot_re, pad_spec)], axis=2)
        out_im_full = ops.concatenate([ops.pad(out_top_im, pad_spec), mid, ops.pad(out_bot_im, pad_spec)], axis=2)

        out_re_full = ops.transpose(out_re_full, (0, 2, 3, 1))
        out_im_full = ops.transpose(out_im_full, (0, 2, 3, 1))
        return irfft2_hw(out_re_full, out_im_full, H, W)


class FNOBlock(layers.Layer):

    def __init__(self, width, modes1, modes2, activation=True, **kwargs):
        super().__init__(**kwargs)
        self.spectral = SpectralConv2D(width, width, modes1, modes2, name="spectral")
        self.pointwise = layers.Conv2D(width, 1, name="pointwise")
        self.act = layers.Activation("gelu") if activation else None

    def call(self, x):
        out = self.spectral(x) + self.pointwise(x)
        return self.act(out) if self.act is not None else out


def FNO2D(input_shape=(64, 64, 1), out_channels=1, width=32, modes1=12, modes2=12,
          num_layers=4, add_grid=True, name="fno2d"):
    H, W = input_shape[0], input_shape[1]
    inputs = keras.Input(shape=input_shape, name="field")
    x = CoordinateGrid(H, W, name="coord_grid")(inputs) if add_grid else inputs

    x = layers.Dense(width, name="fc0")(x)
    for i in range(num_layers):
        x = FNOBlock(width, modes1, modes2, activation=(i < num_layers - 1), name=f"blocks{i}")(x)

    x = layers.Dense(128, activation="gelu", name="fc1")(x)
    x = layers.Dense(out_channels, name="fc2")(x)

    return keras.Model(inputs, x, name=name)
