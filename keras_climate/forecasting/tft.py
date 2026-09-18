"""
keras_climate.forecasting.tft
---------------------------------
Temporal Fusion Transformer (Lim et al. 2021): LSTM encoder-decoder for
local processing, Gated Residual Networks for variable selection and
static-covariate enrichment, and interpretable multi-head attention over
the full lookback+horizon for long-range dependencies. A strong choice for
multi-horizon forecasting with known future inputs (e.g. forecast weather
drivers) and static metadata (e.g. station location/elevation).
"""

import keras
from keras import layers, ops
from keras_climate.utils.layers import GatedResidualNetwork


class VariableSelectionNetwork(layers.Layer):
    """Learns per-timestep softmax weights over input variables and returns
    a single fused embedding per timestep, along with the selection weights
    (useful for interpretability)."""

    def __init__(self, num_vars, hidden_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.var_grns = [GatedResidualNetwork(hidden_dim, dropout, name=f"var_grn{i}")
                          for i in range(num_vars)]
        self.flatten_grn = GatedResidualNetwork(num_vars, dropout, name="flatten_grn")

    def call(self, per_var_embeddings, training=False):
        # per_var_embeddings: list of (B, T, hidden_dim), length num_vars
        flat = ops.concatenate(per_var_embeddings, axis=-1)  # (B, T, num_vars*hidden_dim)
        weights = self.flatten_grn(flat, training=training)  # (B, T, num_vars)
        weights = ops.softmax(weights, axis=-1)

        processed = [grn(v, training=training) for grn, v in zip(self.var_grns, per_var_embeddings)]
        stacked = ops.stack(processed, axis=-2)  # (B, T, num_vars, hidden_dim)
        fused = ops.sum(stacked * weights[..., None], axis=-2)
        return fused, weights


class InterpretableMultiHeadAttention(layers.Layer):
    """Shares the value projection across heads (as in the TFT paper), which
    aids interpretability of attention weights."""

    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.q_layers = [layers.Dense(self.head_dim) for _ in range(num_heads)]
        self.k_layers = [layers.Dense(self.head_dim) for _ in range(num_heads)]
        self.v_layer = layers.Dense(self.head_dim)  # shared value projection
        self.attn_drop = layers.Dropout(dropout)
        self.out_proj = layers.Dense(d_model)

    def call(self, x, mask=None, training=False):
        v = self.v_layer(x)
        head_outs = []
        for h in range(self.num_heads):
            q = self.q_layers[h](x)
            k = self.k_layers[h](x)
            scores = ops.matmul(q, ops.transpose(k, (0, 2, 1))) / (self.head_dim ** 0.5)
            if mask is not None:
                scores = scores + mask
            attn = ops.softmax(scores, axis=-1)
            attn = self.attn_drop(attn, training=training)
            head_outs.append(ops.matmul(attn, v))
        out = ops.mean(ops.stack(head_outs, axis=-1), axis=-1)
        return self.out_proj(out)


def causal_mask(seq_len):
    mask = ops.triu(ops.ones((seq_len, seq_len)) * -1e9, k=1)
    return mask[None, :, :]


def TemporalFusionTransformer(
    encoder_len=168,
    decoder_len=24,
    num_past_vars=6,
    num_future_vars=3,
    num_static_vars=2,
    hidden_dim=64,
    num_heads=4,
    dropout=0.1,
    quantiles=(0.1, 0.5, 0.9),
    name="tft",
):
    """Inputs:
        past_inputs:   (B, encoder_len, num_past_vars)   - historical observed drivers
        future_inputs: (B, decoder_len, num_future_vars)  - known future drivers (e.g. NWP forecast)
        static_inputs: (B, num_static_vars)               - static metadata (station id/location/etc)
    Output: (B, decoder_len, len(quantiles)) quantile forecasts.
    """
    past_in = keras.Input((encoder_len, num_past_vars), name="past_inputs")
    future_in = keras.Input((decoder_len, num_future_vars), name="future_inputs")
    static_in = keras.Input((num_static_vars,), name="static_inputs")

    # --- static covariate encoder ---
    static_embed = layers.Dense(hidden_dim, name="static_embed")(static_in)
    static_context_selection = GatedResidualNetwork(hidden_dim, dropout, name="static_ctx_selection")(static_embed)
    static_context_enrichment = GatedResidualNetwork(hidden_dim, dropout, name="static_ctx_enrichment")(static_embed)
    static_context_h = GatedResidualNetwork(hidden_dim, dropout, name="static_ctx_h")(static_embed)
    static_context_c = GatedResidualNetwork(hidden_dim, dropout, name="static_ctx_c")(static_embed)

    # --- variable selection (encoder) ---
    past_var_embeds = [
        layers.Dense(hidden_dim, name=f"past_var{i}_embed")(
            layers.Lambda(lambda t, idx=i: t[..., idx:idx + 1])(past_in))
        for i in range(num_past_vars)
    ]
    past_selected, past_weights = VariableSelectionNetwork(
        num_past_vars, hidden_dim, dropout, name="past_vsn")(past_var_embeds)

    future_var_embeds = [
        layers.Dense(hidden_dim, name=f"future_var{i}_embed")(
            layers.Lambda(lambda t, idx=i: t[..., idx:idx + 1])(future_in))
        for i in range(num_future_vars)
    ]
    future_selected, future_weights = VariableSelectionNetwork(
        num_future_vars, hidden_dim, dropout, name="future_vsn")(future_var_embeds)

    # --- LSTM encoder/decoder (locally processes each timestep) ---
    encoder_out, state_h, state_c = layers.LSTM(
        hidden_dim, return_sequences=True, return_state=True, name="lstm_encoder"
    )(past_selected, initial_state=[static_context_h, static_context_c])

    decoder_out = layers.LSTM(
        hidden_dim, return_sequences=True, name="lstm_decoder"
    )(future_selected, initial_state=[state_h, state_c])

    lstm_out = layers.Concatenate(axis=1, name="concat_encoder_decoder")([encoder_out, decoder_out])
    lstm_in = layers.Concatenate(axis=1, name="concat_selected")([past_selected, future_selected])

    # gate + add + norm (post-LSTM skip connection)
    gated = GatedResidualNetwork(hidden_dim, dropout, name="post_lstm_gate")(lstm_out)
    x = layers.LayerNormalization(name="post_lstm_norm")(gated + lstm_in)

    # --- static enrichment ---
    enriched = GatedResidualNetwork(hidden_dim, dropout, name="static_enrichment")(
        x + static_context_enrichment[:, None, :]
    )

    # --- interpretable multi-head self-attention over encoder+decoder span ---
    total_len = encoder_len + decoder_len
    mask = causal_mask(total_len)
    attn_out = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout, name="attention")(
        enriched, mask=mask
    )
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.LayerNormalization(name="post_attn_norm")(attn_out + enriched)

    # --- position-wise feed-forward + final skip from pre-enrichment ---
    x = GatedResidualNetwork(hidden_dim, dropout, name="positionwise_ff")(x)
    x = layers.LayerNormalization(name="final_norm")(x + lstm_out)

    # keep only the decoder (forecast horizon) span
    decoder_span = layers.Lambda(lambda t, L=encoder_len: t[:, L:, :], name="slice_decoder_span")(x)

    outputs = layers.Dense(len(quantiles), name="quantile_head")(decoder_span)

    return keras.Model([past_in, future_in, static_in], outputs, name=name)
