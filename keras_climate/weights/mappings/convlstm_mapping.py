import numpy as np


def convert_convlstm_stack_state_dict(flat_state_dict, layer_specs, keras_prefix="encoder_convlstm"):
    out = {}
    for i, (in_ch, hidden_ch) in enumerate(layer_specs):
        w = flat_state_dict[f"cell_list.{i}.conv.weight"]
        b = flat_state_dict[f"cell_list.{i}.conv.bias"]

        gi, gf, go, gg = np.split(w, 4, axis=0)
        w_reordered = np.concatenate([gi, gf, gg, go], axis=0)
        bi, bf, bo, bg = np.split(b, 4, axis=0)
        b_reordered = np.concatenate([bi, bf, bg, bo], axis=0)

        w_input = w_reordered[:, :in_ch]
        w_recurrent = w_reordered[:, in_ch:in_ch + hidden_ch]

        kernel = np.transpose(w_input, (2, 3, 1, 0))
        recurrent_kernel = np.transpose(w_recurrent, (2, 3, 1, 0))

        prefix = f"{keras_prefix}{i}/conv_lstm_cell"
        out[f"{prefix}/kernel"] = kernel
        out[f"{prefix}/recurrent_kernel"] = recurrent_kernel
        out[f"{prefix}/bias"] = b_reordered

    return out


def build_convlstm_identity_mapper():
    return lambda k: k
