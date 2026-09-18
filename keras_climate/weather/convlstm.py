"""
keras_climate.weather.convlstm
----------------------------------
ConvLSTM (Shi et al. 2015): stacked ConvLSTM2D encoder-forecaster for
spatiotemporal nowcasting (radar reflectivity, precipitation, cloud fields).
Uses Keras's built-in ConvLSTM2D cells.
"""

import keras
from keras import layers, ops


def ConvLSTMNowcaster(
    input_shape=(10, 128, 128, 1),  # (T_in, H, W, C)
    pred_steps=10,
    filters=(64, 64, 64),
    kernel_size=3,
    out_channels=1,
    name="convlstm_nowcaster",
):
    """Encoder-forecaster ConvLSTM. Encodes the input sequence with stacked
    ConvLSTM2D layers, then unrolls `pred_steps` future frames autoregressively:
    each predicted frame (in pixel/output-channel space) is fed back as the
    next step's input, through the *same* decoder ConvLSTM2D layers reused
    across every step (a recurrent decoder is only recurrent if its weights
    are shared across time - a fresh layer per step would silently multiply
    the parameter count by `pred_steps` and prevent the model from learning
    a single, time-shared dynamical update rule)."""

    inputs = keras.Input(shape=input_shape, name="frames")
    x = inputs

    states = []
    for i, f in enumerate(filters):
        x, h, c = layers.ConvLSTM2D(
            f, kernel_size, padding="same", return_sequences=True,
            return_state=True, name=f"encoder_convlstm{i}",
        )(x)
        x = layers.BatchNormalization(name=f"encoder_bn{i}")(x)
        states.append([h, c])

    # Decoder cells + read-out head: created once, reused at every rollout step.
    decoder_convlstms = [
        layers.ConvLSTM2D(f, kernel_size, padding="same", return_sequences=True,
                           return_state=True, name=f"decoder_convlstm{i}")
        for i, f in enumerate(filters)
    ]
    decoder_bns = [layers.BatchNormalization(name=f"decoder_bn{i}") for i in range(len(filters))]
    head = layers.Conv3D(out_channels, 1, activation="sigmoid", name="frame_head")

    # Seed the rollout with a zero frame in pixel/output-channel space, so
    # every step (including the first) feeds the shared decoder layers a
    # consistently-shaped input - matching what every later step feeds back
    # (the previous predicted frame), rather than mixing an abstract
    # encoder-feature-space seed with pixel-space frames thereafter.
    H, W = input_shape[1], input_shape[2]
    cur = layers.Lambda(
        lambda t: ops.zeros((ops.shape(t)[0], 1, H, W, out_channels), dtype=t.dtype),
        name="zero_seed",
    )(inputs)

    outputs = []
    cur_states = states
    for step in range(pred_steps):
        new_states = []
        h = cur
        for i in range(len(filters)):
            h, hs, cs = decoder_convlstms[i](h, initial_state=cur_states[i])
            h = decoder_bns[i](h)
            new_states.append([hs, cs])
        frame = head(h)
        outputs.append(frame)
        cur = frame
        cur_states = new_states

    out = layers.Concatenate(axis=1, name="stack_frames")(outputs)
    return keras.Model(inputs, out, name=name)


def convlstm_config(variant="base"):
    presets = {
        "small": dict(filters=(32, 32)),
        "base": dict(filters=(64, 64, 64)),
        "large": dict(filters=(128, 96, 64)),
    }
    return presets[variant]
