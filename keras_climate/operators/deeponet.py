import keras
from keras import layers


class ScalarBias(layers.Layer):

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, x):
        return x + self.bias


def DeepONet(num_sensors=100, coord_dim=2, branch_units=(128, 128, 100),
             trunk_units=(128, 128, 100), activation="relu", name="deeponet"):
    assert branch_units[-1] == trunk_units[-1], "branch/trunk nets must end at the same width"

    branch_in = keras.Input(shape=(num_sensors,), name="branch_input")
    trunk_in = keras.Input(shape=(coord_dim,), name="trunk_input")

    b = branch_in
    for i, u in enumerate(branch_units):
        is_last = i == len(branch_units) - 1
        b = layers.Dense(u, activation=None if is_last else activation, name=f"branch_fc{i}")(b)

    t = trunk_in
    for i, u in enumerate(trunk_units):
        t = layers.Dense(u, activation=activation, name=f"trunk_fc{i}")(t)

    out = layers.Dot(axes=1, name="dot")([b, t])
    out = ScalarBias(name="bias_layer")(out)

    return keras.Model([branch_in, trunk_in], out, name=name)
