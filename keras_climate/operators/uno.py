import keras
from keras import layers
from keras_climate.operators.fno import SpectralConv2D


class UNOBlock(layers.Layer):

    def __init__(self, out_channels, modes1, modes2, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.spectral = SpectralConv2D(in_channels, self.out_channels, self.modes1, self.modes2,
                                        name="spectral")
        self.pointwise = layers.Conv2D(self.out_channels, 1, name="pointwise")
        self.act = layers.Activation("gelu")
        super().build(input_shape)

    def call(self, x):
        return self.act(self.spectral(x) + self.pointwise(x))


def UNO(input_shape=(64, 64, 1), out_channels=1, base_width=16, depth=3, modes1=8, modes2=8,
        name="uno"):
    inputs = keras.Input(shape=input_shape, name="field")
    x = layers.Dense(base_width, name="lift")(inputs)

    widths = [base_width * (2 ** i) for i in range(depth + 1)]
    skips = []
    for i in range(depth):
        x = UNOBlock(widths[i], modes1, modes2, name=f"enc_block{i}")(x)
        skips.append(x)
        x = layers.AveragePooling2D(2, name=f"down{i}")(x)

    x = UNOBlock(widths[depth], modes1, modes2, name="bottleneck")(x)

    for i in reversed(range(depth)):
        x = layers.UpSampling2D(2, interpolation="bilinear", name=f"up{i}")(x)
        x = layers.Concatenate(name=f"concat{i}")([x, skips[i]])
        x = UNOBlock(widths[i], modes1, modes2, name=f"dec_block{i}")(x)

    out = layers.Dense(out_channels, name="project")(x)
    return keras.Model(inputs, out, name=name)
