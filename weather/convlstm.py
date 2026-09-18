"""
keras_climate.weather.convlstm
----------------------------------
ConvLSTM (Shi et al. 2015): stacked ConvLSTM2D encoder-forecaster for
spatiotemporal nowcasting (radar reflectivity, precipitation, cloud fields).
Uses Keras's built-in ConvLSTM2D cells.
"""

import keras
from keras import layers


def ConvLSTMNowcaster(
    input_shape=(10, 128, 128, 1),  # (T_in, H, W, C)
    pred_steps=10,
    filters=(64, 64, 64),
    kernel_size=3,
    out_channels=1,
    name="convlstm_nowcaster",
):
    """Encoder-forecaster ConvLSTM. Encodes the input sequence with stacked
    ConvLSTM2D layers, then unrolls `pred_steps` future frames by feeding
    the last hidden state's prediction back as the next input (teacher-free
    autoregressive rollout at inference; for training you may instead
    supply teacher-forced targets by wrapping this model)."""

    inputs = keras.Input(shape=input_shape, name="frames")
    x = inputs

    states = []
    for i, f in enumerate(filters):
        return_seq = True
        x, h, c = layers.ConvLSTM2D(
            f, kernel_size, padding="same", return_sequences=return_seq,
            return_state=True, name=f"encoder_convlstm{i}",
        )(x)
        x = layers.BatchNormalization(name=f"encoder_bn{i}")(x)
        states.append((h, c))

    # Take the last encoded frame as the seed for autoregressive rollout.
    last_frame = layers.Lambda(lambda t: t[:, -1:, :, :, :], name="last_frame")(x)

    outputs = []
    cur = last_frame
    cur_states = states
    for step in range(pred_steps):
        new_states = []
        h = cur
        for i, f in enumerate(filters):
            h, hs, cs = layers.ConvLSTM2D(
                f, kernel_size, padding="same", return_sequences=True,
                return_state=True, name=f"decoder_convlstm{i}_step{step}",
            )(h, initial_state=cur_states[i])
            h = layers.BatchNormalization(name=f"decoder_bn{i}_step{step}")(h)
            new_states.append((hs, cs))
        frame = layers.Conv3D(out_channels, 1, activation="sigmoid",
                               name=f"frame_head_step{step}")(h)
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
