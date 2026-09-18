"""
Mapping for `keras_climate.weather.convlstm.ConvLSTMNowcaster`, targeting
the widely-used `ndrplz/ConvLSTM_pytorch` reference cell implementation's
naming convention (`cell_list.{i}.conv.{weight,bias}`).

No single "original ConvLSTMNowcaster" checkpoint exists publicly - the
1embed_dim5 Shi et al. 2015 paper never released weights, and
`ndrplz/ConvLSTM_pytorch` (far and away the most widely used open-source
reference) provides only the recurrent *cell*, not a full encoder-decoder
forecaster - different projects build very different scaffolding on top of
it. This mapper therefore targets the cell level: it ports a stack of
`ConvLSTMCell`s onto either this model's encoder *or* decoder stack (they
share the same cell architecture, just independently-learned weights),
which is the well-defined, checkable unit of portability here.

Getting this right requires two real layout differences, not just a
transpose:
  * ndrplz fuses the input-conv and recurrent-conv into a single
    `nn.Conv2d` over the channel-concatenation `[inputs, h_prev]` - Keras's
    `ConvLSTM2D` keeps them as two separate weights (`kernel`,
    `recurrent_kernel`), so the fused kernel must be *split* along its
    input-channel axis, not just reshaped.
  * The two implementations concatenate the four gates in a different
    order: ndrplz uses (input, forget, output, candidate); Keras's
    `ConvLSTMCell.call` (verified directly from source) uses (input,
    forget, candidate, output). The gate blocks must be *reordered*, not
    just relabeled.
"""

import numpy as np


def convert_convlstm_stack_state_dict(flat_state_dict, layer_specs, keras_prefix="encoder_convlstm"):
    """
    Args:
        flat_state_dict: `{"cell_list.{i}.conv.weight": ndarray, "cell_list.{i}.conv.bias": ndarray, ...}`
            (ndrplz `ConvLSTM`'s state_dict, one entry pair per stacked cell).
        layer_specs: list of `(in_channels, hidden_channels)` per stacked
            layer, in encoder/decoder order.
        keras_prefix: e.g. `"encoder_convlstm"` or `"decoder_convlstm"` -
            matching `ConvLSTMNowcaster`'s per-stage layer names (layer `i`
            -> `f"{keras_prefix}{i}"`).

    Returns `{keras_weight_path: np.ndarray}`, ready to assign via an
    identity mapper (see `build_convlstm_identity_mapper`).
    """
    out = {}
    for i, (in_ch, hidden_ch) in enumerate(layer_specs):
        w = flat_state_dict[f"cell_list.{i}.conv.weight"]  # (4*hidden, in+hidden, kh, kw)
        b = flat_state_dict[f"cell_list.{i}.conv.bias"]    # (4*hidden,)

        # ndrplz gate order (i, f, o, g) -> Keras gate order (i, f, c, o)
        # (g/candidate and o/output swap positions).
        gi, gf, go, gg = np.split(w, 4, axis=0)
        w_reordered = np.concatenate([gi, gf, gg, go], axis=0)
        bi, bf, bo, bg = np.split(b, 4, axis=0)
        b_reordered = np.concatenate([bi, bf, bg, bo], axis=0)

        # Split the fused [input, h_prev] conv into Keras's separate
        # input-kernel / recurrent-kernel along the input-channel axis.
        w_input = w_reordered[:, :in_ch]
        w_recurrent = w_reordered[:, in_ch:in_ch + hidden_ch]

        kernel = np.transpose(w_input, (2, 3, 1, 0))          # (kh, kw, in_ch, 4*hidden)
        recurrent_kernel = np.transpose(w_recurrent, (2, 3, 1, 0))  # (kh, kw, hidden_ch, 4*hidden)

        prefix = f"{keras_prefix}{i}/conv_lstm_cell"
        out[f"{prefix}/kernel"] = kernel
        out[f"{prefix}/recurrent_kernel"] = recurrent_kernel
        out[f"{prefix}/bias"] = b_reordered

    return out


def build_convlstm_identity_mapper():
    """`convert_convlstm_stack_state_dict` already produces keys spelled
    exactly as the target Keras weight paths."""
    return lambda k: k
