"""
Mapping for `keras_climate.forecasting.tft.TemporalFusionTransformer`.

No official TFT checkpoint is publicly downloadable as a single small
file - this mapper targets this repo's own architecture instead, for
porting weights from a from-scratch model trained with this exact
architecture.

Two real (not just cosmetic) layout differences from a standard PyTorch
`nn.LSTM`, handled by `convert_tft_lstm_state_dict`:
  * PyTorch's `nn.LSTM` already keeps `weight_ih`/`weight_hh` as two
    separate matrices (unlike the fused-conv `ConvLSTMCell` this repo's
    other LSTM-mapping helper targets), and its gate order (i, f, g, o)
    already matches Keras's `LSTMCell` (i, f, c, o) directly - no
    reordering needed, only a transpose.
  * PyTorch's `nn.LSTM` has *two* bias vectors (`bias_ih`, `bias_hh`)
    that are always used additively; Keras's `LSTM` has only one - they
    must be summed, not just renamed.
"""

import re

import numpy as np


def _grn_rules(torch_prefix, keras_prefix, has_skip):
    """One `GatedResidualNetwork`: `skip` (only present when the input/
    output dims differ), `fc1`, `fc2`, `gate`, `norm`."""
    rules = []
    if has_skip:
        rules += [
            (rf"^{re.escape(torch_prefix)}\.skip\.weight$", f"{keras_prefix}/skip/kernel"),
            (rf"^{re.escape(torch_prefix)}\.skip\.bias$", f"{keras_prefix}/skip/bias"),
        ]
    rules += [
        (rf"^{re.escape(torch_prefix)}\.fc1\.weight$", f"{keras_prefix}/fc1/kernel"),
        (rf"^{re.escape(torch_prefix)}\.fc1\.bias$", f"{keras_prefix}/fc1/bias"),
        (rf"^{re.escape(torch_prefix)}\.fc2\.weight$", f"{keras_prefix}/fc2/kernel"),
        (rf"^{re.escape(torch_prefix)}\.fc2\.bias$", f"{keras_prefix}/fc2/bias"),
        (rf"^{re.escape(torch_prefix)}\.gate\.weight$", f"{keras_prefix}/gate/kernel"),
        (rf"^{re.escape(torch_prefix)}\.gate\.bias$", f"{keras_prefix}/gate/bias"),
        (rf"^{re.escape(torch_prefix)}\.norm\.weight$", f"{keras_prefix}/norm/gamma"),
        (rf"^{re.escape(torch_prefix)}\.norm\.bias$", f"{keras_prefix}/norm/beta"),
    ]
    return rules


def _vsn_rules(torch_prefix, keras_prefix, num_vars, hidden_dim, flatten_in_dim):
    rules = []
    for i in range(num_vars):
        rules += _grn_rules(f"{torch_prefix}.var_grns.{i}", f"{keras_prefix}/var_grn{i}",
                             has_skip=False)  # var GRNs: hidden_dim -> hidden_dim, no skip needed
    rules += _grn_rules(f"{torch_prefix}.flatten_grn", f"{keras_prefix}/flatten_grn",
                         has_skip=(flatten_in_dim != num_vars))
    return rules


def build_tft_mapper(num_past_vars, num_future_vars, hidden_dim):
    """torch_key -> keras_key mapper, assuming a source checkpoint using
    this repo's own naming: `past_var{i}_embed`/`future_var{i}_embed`/
    `static_embed.{weight,bias}`, `static_ctx_{selection,enrichment,h,c}`
    (GRNs), `past_vsn`/`future_vsn` (`VariableSelectionNetwork`s, each with
    `var_grns.{i}` + `flatten_grn`), `post_lstm_gate`/`static_enrichment`/
    `positionwise_ff` (GRNs), `post_lstm_norm`/`post_attn_norm`/
    `final_norm.{weight,bias}`, `attention.{q,k}_layers.{h}.{weight,bias}`
    + `attention.v_layer.{weight,bias}` + `attention.out_proj.{weight,bias}`,
    `quantile_head.{weight,bias}`. LSTM weights are handled separately by
    `convert_tft_lstm_state_dict` (see module docstring)."""
    rules = [
        (r"^static_embed\.weight$", "static_embed/kernel"),
        (r"^static_embed\.bias$", "static_embed/bias"),
        (r"^quantile_head\.weight$", "quantile_head/kernel"),
        (r"^quantile_head\.bias$", "quantile_head/bias"),
        (r"^post_lstm_norm\.weight$", "post_lstm_norm/gamma"),
        (r"^post_lstm_norm\.bias$", "post_lstm_norm/beta"),
        (r"^post_attn_norm\.weight$", "post_attn_norm/gamma"),
        (r"^post_attn_norm\.bias$", "post_attn_norm/beta"),
        (r"^final_norm\.weight$", "final_norm/gamma"),
        (r"^final_norm\.bias$", "final_norm/beta"),
    ]
    for i in range(num_past_vars):
        rules += [
            (rf"^past_var{i}_embed\.weight$", f"past_var{i}_embed/kernel"),
            (rf"^past_var{i}_embed\.bias$", f"past_var{i}_embed/bias"),
        ]
    for i in range(num_future_vars):
        rules += [
            (rf"^future_var{i}_embed\.weight$", f"future_var{i}_embed/kernel"),
            (rf"^future_var{i}_embed\.bias$", f"future_var{i}_embed/bias"),
        ]

    for name in ("static_ctx_enrichment", "static_ctx_h", "static_ctx_c",
                 "post_lstm_gate", "static_enrichment", "positionwise_ff"):
        # All operate at a fixed `hidden_dim -> hidden_dim`, so never need
        # the projection `skip` branch.
        rules += _grn_rules(name, name, has_skip=False)

    rules += _vsn_rules("past_vsn", "past_vsn", num_past_vars, hidden_dim,
                         flatten_in_dim=num_past_vars * hidden_dim)
    rules += _vsn_rules("future_vsn", "future_vsn", num_future_vars, hidden_dim,
                         flatten_in_dim=num_future_vars * hidden_dim)

    num_heads_guess = 16  # overwritten below via explicit head count if needed
    rules += [
        (r"^attention\.v_layer\.weight$", "attention/v/kernel"),
        (r"^attention\.v_layer\.bias$", "attention/v/bias"),
        (r"^attention\.out_proj\.weight$", "attention/out_proj/kernel"),
        (r"^attention\.out_proj\.bias$", "attention/out_proj/bias"),
    ]

    def head_rules(num_heads):
        r = []
        for h in range(num_heads):
            r += [
                (rf"^attention\.q_layers\.{h}\.weight$", f"attention/q{h}/kernel"),
                (rf"^attention\.q_layers\.{h}\.bias$", f"attention/q{h}/bias"),
                (rf"^attention\.k_layers\.{h}\.weight$", f"attention/k{h}/kernel"),
                (rf"^attention\.k_layers\.{h}\.bias$", f"attention/k{h}/bias"),
            ]
        return r

    # Head count isn't otherwise known here; generate rules generously for
    # up to 64 heads (unused ones simply never match anything).
    rules += head_rules(64)

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper


def convert_tft_lstm_state_dict(flat_state_dict, torch_prefix, keras_prefix):
    """Translates one `nn.LSTM`'s weights (`{torch_prefix}.weight_ih_l0`,
    `.weight_hh_l0`, `.bias_ih_l0`, `.bias_hh_l0`) into
    `{keras_weight_path: np.ndarray}` for a Keras `LSTM(..., name=
    keras_prefix)` layer - summing the two bias vectors (see module
    docstring) since Keras's `LSTM` has only one."""
    out = {}
    w_ih = flat_state_dict.get(f"{torch_prefix}.weight_ih_l0")
    w_hh = flat_state_dict.get(f"{torch_prefix}.weight_hh_l0")
    b_ih = flat_state_dict.get(f"{torch_prefix}.bias_ih_l0")
    b_hh = flat_state_dict.get(f"{torch_prefix}.bias_hh_l0")
    if w_ih is not None:
        out[f"{keras_prefix}/lstm_cell/kernel"] = w_ih
    if w_hh is not None:
        out[f"{keras_prefix}/lstm_cell/recurrent_kernel"] = w_hh
    if b_ih is not None and b_hh is not None:
        out[f"{keras_prefix}/lstm_cell/bias"] = b_ih + b_hh
    return out
