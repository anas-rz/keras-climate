"""Build/shape sanity checks + PyTorch weight-port round-trip test for
`keras_climate.forecasting.tft`.

Run with: pytest keras_climate/forecasting/test_tft.py
"""
import numpy as np
import pytest
import keras

from keras_climate.forecasting.tft import TemporalFusionTransformer, GatedResidualNetwork
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_tft_mapper, convert_tft_lstm_state_dict


# --------------------------------------------------------------------------
# Build / forward-pass sanity checks (Keras only, no torch required)
# --------------------------------------------------------------------------

def test_builds_and_runs():
    model = TemporalFusionTransformer(encoder_len=24, decoder_len=6, num_past_vars=3,
                                       num_future_vars=2, num_static_vars=2, hidden_dim=16,
                                       num_heads=2, quantiles=(0.1, 0.5, 0.9))
    past = np.random.randn(2, 24, 3).astype("float32")
    future = np.random.randn(2, 6, 2).astype("float32")
    static = np.random.randn(2, 2).astype("float32")
    y = keras.ops.convert_to_numpy(model([past, future, static]))
    assert y.shape == (2, 6, 3)


def test_grn_sublayers_are_named_deterministically():
    """Regression check: GatedResidualNetwork's internal Dense/LayerNorm
    sublayers must have fixed names, not Keras's globally-incrementing
    auto-names (which would depend on how many other unnamed layers exist
    elsewhere in the process - see utils/layers.py's GatedResidualNetwork)."""
    grn = GatedResidualNetwork(units=8, name="my_grn")
    grn(np.zeros((1, 8), dtype="float32"))  # build() alone doesn't materialize sublayer weights
    names = {w.path.split("/")[-2] for w in grn.weights}
    assert names == {"fc1", "fc2", "gate", "norm"}


def test_grn_skip_projection_only_when_dims_differ():
    grn_same = GatedResidualNetwork(units=8)
    grn_same(np.zeros((1, 8), dtype="float32"))
    assert grn_same.skip is None

    grn_diff = GatedResidualNetwork(units=8)
    grn_diff(np.zeros((1, 4), dtype="float32"))
    assert grn_diff.skip is not None


# --------------------------------------------------------------------------
# PyTorch weight-port round-trip against a from-scratch reference of this
# repo's own architecture (see weights/mappings/tft_mapping.py's module
# docstring: no directly-downloadable official checkpoint exists).
# --------------------------------------------------------------------------

def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class TorchGRN(nn.Module):
        def __init__(self, in_dim, units):
            super().__init__()
            self.skip = nn.Linear(in_dim, units) if in_dim != units else None
            self.fc1 = nn.Linear(in_dim, units)
            self.fc2 = nn.Linear(units, units)
            self.gate = nn.Linear(units, units * 2)
            self.norm = nn.LayerNorm(units)

        def forward(self, x):
            skip = self.skip(x) if self.skip is not None else x
            h = self.fc2(torch.nn.functional.elu(self.fc1(x)))
            value, gate = self.gate(h).chunk(2, dim=-1)
            h = value * torch.sigmoid(gate)
            return self.norm(skip + h)

    class TorchVSN(nn.Module):
        def __init__(self, num_vars, hidden_dim):
            super().__init__()
            self.var_grns = nn.ModuleList([TorchGRN(hidden_dim, hidden_dim) for _ in range(num_vars)])
            self.flatten_grn = TorchGRN(num_vars * hidden_dim, num_vars)

        def forward(self, per_var_embeds):
            flat = torch.cat(per_var_embeds, dim=-1)
            weights = torch.softmax(self.flatten_grn(flat), dim=-1)
            processed = torch.stack([grn(v) for grn, v in zip(self.var_grns, per_var_embeds)], dim=-2)
            return (processed * weights.unsqueeze(-1)).sum(dim=-2)

    class TorchAttention(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.q_layers = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
            self.k_layers = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
            self.v_layer = nn.Linear(d_model, self.head_dim)
            self.out_proj = nn.Linear(self.head_dim, d_model)

        def forward(self, x, mask):
            v = self.v_layer(x)
            head_outs = []
            for q_l, k_l in zip(self.q_layers, self.k_layers):
                q, k = q_l(x), k_l(x)
                scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) + mask
                attn = torch.softmax(scores, dim=-1)
                head_outs.append(attn @ v)
            out = torch.stack(head_outs, dim=-1).mean(dim=-1)
            return self.out_proj(out)

    class TorchTFT(nn.Module):
        def __init__(self, encoder_len, decoder_len, num_past_vars, num_future_vars,
                     num_static_vars, hidden_dim, num_heads, num_quantiles):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.static_embed = nn.Linear(num_static_vars, hidden_dim)
            self.static_ctx_enrichment = TorchGRN(hidden_dim, hidden_dim)
            self.static_ctx_h = TorchGRN(hidden_dim, hidden_dim)
            self.static_ctx_c = TorchGRN(hidden_dim, hidden_dim)

            # Individually-named attributes (not a ModuleList) to match the
            # mapper's `past_var{i}_embed`/`future_var{i}_embed` naming
            # convention exactly (a ModuleList would produce
            # `past_var_embeds.{i}.weight` instead).
            for i in range(num_past_vars):
                setattr(self, f"past_var{i}_embed", nn.Linear(1, hidden_dim))
            self.num_past_vars = num_past_vars
            self.past_vsn = TorchVSN(num_past_vars, hidden_dim)
            for i in range(num_future_vars):
                setattr(self, f"future_var{i}_embed", nn.Linear(1, hidden_dim))
            self.num_future_vars = num_future_vars
            self.future_vsn = TorchVSN(num_future_vars, hidden_dim)

            self.lstm_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.lstm_decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

            self.post_lstm_gate = TorchGRN(hidden_dim, hidden_dim)
            self.post_lstm_norm = nn.LayerNorm(hidden_dim)
            self.static_enrichment = TorchGRN(hidden_dim, hidden_dim)
            self.attention = TorchAttention(hidden_dim, num_heads)
            self.post_attn_norm = nn.LayerNorm(hidden_dim)
            self.positionwise_ff = TorchGRN(hidden_dim, hidden_dim)
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.quantile_head = nn.Linear(hidden_dim, num_quantiles)

            self.encoder_len, self.decoder_len = encoder_len, decoder_len

        def forward(self, past_in, future_in, static_in):
            static_embed = self.static_embed(static_in)
            static_ctx_enrichment = self.static_ctx_enrichment(static_embed)
            static_ctx_h = self.static_ctx_h(static_embed)
            static_ctx_c = self.static_ctx_c(static_embed)

            past_embeds = [getattr(self, f"past_var{i}_embed")(past_in[..., i:i + 1])
                           for i in range(self.num_past_vars)]
            past_selected = self.past_vsn(past_embeds)
            future_embeds = [getattr(self, f"future_var{i}_embed")(future_in[..., i:i + 1])
                             for i in range(self.num_future_vars)]
            future_selected = self.future_vsn(future_embeds)

            h0 = static_ctx_h.unsqueeze(0)
            c0 = static_ctx_c.unsqueeze(0)
            enc_out, (h_n, c_n) = self.lstm_encoder(past_selected, (h0, c0))
            dec_out, _ = self.lstm_decoder(future_selected, (h_n, c_n))

            lstm_out = torch.cat([enc_out, dec_out], dim=1)
            lstm_in = torch.cat([past_selected, future_selected], dim=1)
            gated = self.post_lstm_gate(lstm_out)
            x = self.post_lstm_norm(gated + lstm_in)

            enriched = self.static_enrichment(x + static_ctx_enrichment.unsqueeze(1))

            total_len = self.encoder_len + self.decoder_len
            mask = torch.triu(torch.ones(total_len, total_len) * -1e9, diagonal=1)
            attn_out = self.attention(enriched, mask)
            x = self.post_attn_norm(attn_out + enriched)

            x = self.positionwise_ff(x)
            x = self.final_norm(x + lstm_out)

            decoder_span = x[:, self.encoder_len:, :]
            return self.quantile_head(decoder_span)

    torch.manual_seed(0)
    encoder_len, decoder_len = 8, 4
    num_past_vars, num_future_vars, num_static_vars = 2, 1, 2
    hidden_dim, num_heads, num_quantiles = 8, 2, 3

    torch_model = TorchTFT(encoder_len, decoder_len, num_past_vars, num_future_vars,
                            num_static_vars, hidden_dim, num_heads, num_quantiles)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = TemporalFusionTransformer(
        encoder_len=encoder_len, decoder_len=decoder_len, num_past_vars=num_past_vars,
        num_future_vars=num_future_vars, num_static_vars=num_static_vars, hidden_dim=hidden_dim,
        num_heads=num_heads, dropout=0.0, quantiles=tuple(range(num_quantiles)),
    )

    grn_mapper = build_tft_mapper(num_past_vars=num_past_vars, num_future_vars=num_future_vars,
                                   hidden_dim=hidden_dim)
    lstm_state = {}
    lstm_state.update(convert_tft_lstm_state_dict(state_dict, "lstm_encoder", "lstm_encoder"))
    lstm_state.update(convert_tft_lstm_state_dict(state_dict, "lstm_decoder", "lstm_decoder"))

    # One converter pass covering everything: `lstm_state` is already keyed
    # by target Keras path (an identity lookup), so it's tried first before
    # falling back to the regex-based GRN/embed/attention mapper - avoids
    # running two separate converters each against the *full* model (which
    # would make every weight the other one owns look spuriously "missing").
    # The raw `lstm_{encoder,decoder}.*` torch keys are dropped here since
    # they're superseded by their already-converted `lstm_state` entries -
    # left in, they'd never be matched (nothing maps a keras key back to a
    # *raw* LSTM torch key) and would always show up as "unused".
    non_lstm_state = {k: v for k, v in state_dict.items()
                       if not k.startswith(("lstm_encoder.", "lstm_decoder."))}
    combined_state = {**non_lstm_state, **lstm_state}

    def combined_mapper(key):
        if key in lstm_state:
            return key
        return grn_mapper(key)

    report = WeightConverter(keras_model, combined_state, combined_mapper).convert(
        strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    past_np = np.random.randn(2, encoder_len, num_past_vars).astype("float32")
    future_np = np.random.randn(2, decoder_len, num_future_vars).astype("float32")
    static_np = np.random.randn(2, num_static_vars).astype("float32")

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(past_np), torch.from_numpy(future_np),
                                 torch.from_numpy(static_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(
        keras_model([past_np, future_np, static_np], training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"TFT weight port numerical mismatch: max abs diff {max_diff}"
