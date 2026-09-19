import numpy as np
import pytest
import keras

from keras_climate.weather.convlstm import ConvLSTMNowcaster, convlstm_config
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import (
    convert_convlstm_stack_state_dict,
    build_convlstm_identity_mapper,
)


@pytest.mark.parametrize("variant", ["small", "base", "large"])
def test_builds_and_runs(variant):
    cfg = convlstm_config(variant)
    model = ConvLSTMNowcaster(
        input_shape=(5, 32, 32, 1), pred_steps=4, out_channels=1, **cfg
    )
    x = np.random.randn(1, 5, 32, 32, 1).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (1, 4, 32, 32, 1)
    assert (y >= 0).all() and (y <= 1).all()


def test_decoder_weights_are_shared_across_rollout_steps():
    filters = (8, 8)
    params_by_pred_steps = set()
    for pred_steps in (2, 5, 9):
        model = ConvLSTMNowcaster(
            input_shape=(3, 16, 16, 1), pred_steps=pred_steps, filters=filters
        )
        params_by_pred_steps.add(model.count_params())
    assert (
        len(params_by_pred_steps) == 1
    ), f"parameter count changed with pred_steps: {params_by_pred_steps}"


def test_multichannel_output():
    model = ConvLSTMNowcaster(
        input_shape=(4, 16, 16, 2), pred_steps=3, filters=(8,), out_channels=3
    )
    x = np.random.randn(1, 4, 16, 16, 2).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (1, 3, 16, 16, 3)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class ConvLSTMCell(nn.Module):

        def __init__(self, input_dim, hidden_dim, kernel_size):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            padding = kernel_size // 2
            self.conv = nn.Conv2d(
                input_dim + hidden_dim,
                4 * hidden_dim,
                kernel_size,
                padding=padding,
                bias=True,
            )

        def forward(self, x, cur_state):
            h_cur, c_cur = cur_state
            combined = torch.cat([x, h_cur], dim=1)
            combined_conv = self.conv(combined)
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            c_next = f * c_cur + i * g
            h_next = o * torch.tanh(c_next)
            return h_next, c_next

        def init_hidden(self, batch_size, h, w):
            return (
                torch.zeros(batch_size, self.hidden_dim, h, w),
                torch.zeros(batch_size, self.hidden_dim, h, w),
            )

    class ConvLSTMStack(nn.Module):
        def __init__(self, layer_specs, kernel_size=3):
            super().__init__()
            self.cell_list = nn.ModuleList(
                [
                    ConvLSTMCell(in_dim, hidden_dim, kernel_size)
                    for in_dim, hidden_dim in layer_specs
                ]
            )

        def forward(self, x):
            B, T, _, H, W = x.shape
            states = [cell.init_hidden(B, H, W) for cell in self.cell_list]
            cur_input = x
            for layer_idx, cell in enumerate(self.cell_list):
                h, c = states[layer_idx]
                outputs = []
                for t in range(T):
                    h, c = cell(cur_input[:, t], (h, c))
                    outputs.append(h)
                cur_input = torch.stack(outputs, dim=1)
                states[layer_idx] = (h, c)
            return cur_input, states

    torch.manual_seed(0)
    layer_specs = [(1, 8), (8, 8)]
    torch_model = ConvLSTMStack(layer_specs, kernel_size=3)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    flat = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}
    translated = convert_convlstm_stack_state_dict(
        flat, layer_specs, keras_prefix="encoder_convlstm"
    )

    keras_model = ConvLSTMNowcaster(
        input_shape=(4, 16, 16, 1), pred_steps=1, filters=(8, 8)
    )
    report = WeightConverter(
        keras_model, translated, build_convlstm_identity_mapper()
    ).convert(strict=False, verbose=False)
    unmatched_non_encoder = [
        k for k in report["missing_in_source"] if "encoder_convlstm" in k
    ]
    assert not unmatched_non_encoder, unmatched_non_encoder

    x_np = np.random.randn(1, 4, 1, 16, 16).astype("float32")
    with torch.no_grad():
        _, torch_states = torch_model(torch.from_numpy(x_np))
    torch_h_final = torch_states[-1][0].numpy()

    keras_in = np.transpose(x_np, (0, 1, 3, 4, 2))
    enc_layer1 = keras_model.get_layer("encoder_convlstm1")
    encoder_output = keras.Model(keras_model.input, enc_layer1.output).predict(
        keras_in, verbose=0
    )
    keras_h_final = keras.ops.convert_to_numpy(encoder_output[1])
    keras_h_final = np.transpose(keras_h_final, (0, 3, 1, 2))

    max_diff = np.abs(torch_h_final - keras_h_final).max()
    assert (
        max_diff < 1e-3
    ), f"ConvLSTM weight port numerical mismatch: max abs diff {max_diff}"
