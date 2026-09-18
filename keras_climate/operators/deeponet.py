"""
keras_climate.operators.deeponet
------------------------------------
DeepONet (Lu et al. 2021, "Learning nonlinear operators via DeepONet"):
learns a mapping from an input *function* (sampled at a fixed set of
sensor points, encoded by a "branch net" MLP) to an output function
evaluated at an arbitrary query coordinate (encoded by a "trunk net" MLP)
- the two encodings are combined via a dot product plus a learned scalar
bias. Unlike a grid-based operator (FNO/UNO/AFNO), DeepONet's output can
be queried at any continuous coordinate, not just points on the input's
discretization grid.

A real checkpoint exists (`BGLab/DeepONet-FlowBench-FPO` on HuggingFace),
but it wraps the branch net with a custom multi-scale Inception-style CNN
feature extractor (parallel 1x1/3x3/5x5 conv branches merged via a fusion
conv) whose exact wiring - which conv output feeds which, in what order
the three scales are fused - can't be determined from the checkpoint's
tensor shapes/names alone without its source code, so porting it would
risk assigning real weights into a subtly wrong computation graph. This
module implements the canonical DeepONet (branch net operating directly
on flattened sensor values, no CNN preprocessing) and is validated
against a from-scratch synthetic PyTorch reference of that architecture
(see `weights/mappings/deeponet_mapping.py`).
"""

import keras
from keras import layers


class ScalarBias(layers.Layer):
    """A single learned scalar added to the branch-trunk dot product -
    matches the official DeepONet's output bias term."""

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(), initializer="zeros", trainable=True, name="bias")
        super().build(input_shape)

    def call(self, x):
        return x + self.bias


def DeepONet(num_sensors=100, coord_dim=2, branch_units=(128, 128, 100),
             trunk_units=(128, 128, 100), activation="relu", name="deeponet"):
    """Inputs:
        branch_input: (B, num_sensors) - the input function's values at
            a fixed set of sensor points.
        trunk_input: (B, coord_dim) - the query coordinate to evaluate
            the output function at.
    Output: (B, 1) - the operator's output at that query coordinate.
    `branch_units[-1]` must equal `trunk_units[-1]` (the shared number of
    basis functions the dot product combines).
    """
    assert branch_units[-1] == trunk_units[-1], "branch/trunk nets must end at the same width"

    branch_in = keras.Input(shape=(num_sensors,), name="branch_input")
    trunk_in = keras.Input(shape=(coord_dim,), name="trunk_input")

    b = branch_in
    for i, u in enumerate(branch_units):
        # No activation on the branch net's final layer (official
        # convention); the trunk net's final layer keeps its activation.
        is_last = i == len(branch_units) - 1
        b = layers.Dense(u, activation=None if is_last else activation, name=f"branch_fc{i}")(b)

    t = trunk_in
    for i, u in enumerate(trunk_units):
        t = layers.Dense(u, activation=activation, name=f"trunk_fc{i}")(t)

    out = layers.Dot(axes=1, name="dot")([b, t])  # (B, 1)
    out = ScalarBias(name="bias_layer")(out)

    return keras.Model([branch_in, trunk_in], out, name=name)
