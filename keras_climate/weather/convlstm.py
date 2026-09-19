import keras
from keras import layers, ops


def ConvLSTMNowcaster(
    input_shape=(10, 128, 128, 1),
    pred_steps=10,
    filters=(64, 64, 64),
    kernel_size=3,
    out_channels=1,
    name="convlstm_nowcaster",
):

    inputs = keras.Input(shape=input_shape, name="frames")
    x = inputs

    states = []
    for i, f in enumerate(filters):
        x, h, c = layers.ConvLSTM2D(
            f,
            kernel_size,
            padding="same",
            return_sequences=True,
            return_state=True,
            name=f"encoder_convlstm{i}",
        )(x)
        x = layers.BatchNormalization(name=f"encoder_bn{i}")(x)
        states.append([h, c])

    decoder_convlstms = [
        layers.ConvLSTM2D(
            f,
            kernel_size,
            padding="same",
            return_sequences=True,
            return_state=True,
            name=f"decoder_convlstm{i}",
        )
        for i, f in enumerate(filters)
    ]
    decoder_bns = [
        layers.BatchNormalization(name=f"decoder_bn{i}") for i in range(len(filters))
    ]
    head = layers.Conv3D(out_channels, 1, activation="sigmoid", name="frame_head")

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
