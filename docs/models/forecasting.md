# Forecasting

`keras_climate.forecasting` — multivariate time-series forecasting.

None of the models in this family have a publicly available
general-purpose pretrained checkpoint (the official repos generally only
ship per-benchmark-dataset training scripts/configs, not reusable
backbones) — each is validated against a from-scratch synthetic PyTorch
reference of its own architecture instead. See
[Weight Porting](../weight-porting.md) for how those round-trip tests work.

## PatchTST

Channel-independent patch-based transformer (Nie et al. 2023) for
long-horizon multivariate forecasting. Each variate (e.g. a climate
station's temperature, humidity, pressure channel) is patched and encoded
independently by a shared transformer, then linearly projected to the
forecast horizon.

`padding_patch="end"` (the official default) replicates the last timestep
`stride` times before patching, so the final `stride`-sized tail of the
series still contributes one more patch instead of being dropped. Pass
`padding_patch=None` for the simpler no-padding variant.

## TimesNet

Discovers dominant periodicities in a series via FFT (Wu et al. 2023),
reshapes the 1D series into 2D tensors (period × num_periods) for each
dominant period, applies Inception-style 2D convolutions to capture both
intra- and inter-period variation, then adaptively fuses the multi-period
representations by their softmax-normalized FFT amplitude weight. Well
suited to seasonal climate variables (diurnal / annual cycles).

## TFT (Temporal Fusion Transformer)

LSTM encoder-decoder for local processing (Lim et al. 2021), Gated
Residual Networks for variable selection and static-covariate enrichment,
and interpretable multi-head attention (value projection shared across
heads) over the full lookback+horizon for long-range dependencies. A
strong choice for multi-horizon forecasting with known future inputs (e.g.
forecast weather drivers) and static metadata (e.g. station
location/elevation).

Inputs: `past_inputs` (historical observed drivers), `future_inputs`
(known future drivers, e.g. an NWP forecast), `static_inputs` (station
id/location/etc). Output: quantile forecasts over the decoder horizon.

This module deliberately omits the paper's `static_context_selection`
vector (which would condition the variable-selection GRNs with an extra
context input) — see [Implementation Notes](../notes.md) for why.

## DLinear

Decomposes the input into a trend (moving-average) and a seasonal
(residual) component (Zeng et al. 2022), each passed through its own
single Linear layer over the time axis, then summed — a deliberately
minimal baseline that outperformed several contemporary Transformer
forecasters on common long-horizon benchmarks.

## N-BEATS

A deep stack of fully-connected blocks (Oreshkin et al. 2020). Each block
maps the lookback window to a small set of "theta" coefficients that
parameterize an interpretable basis expansion (generic, polynomial trend,
or Fourier seasonality), producing both a "backcast" (subtracted from the
residual seen by later blocks) and a "forecast" (added into the running
total) — the paper's "doubly residual stacking". Operates on a single
(univariate) series per call; apply per-channel for multivariate series.

## Informer

An efficient long-sequence Transformer forecaster (Zhou et al. 2021, AAAI
Best Paper) built around: "self-attention distilling" (halving the
sequence length between encoder layers via a strided conv, so a deep
encoder stays tractable over long lookback windows), and a generative-style
decoder that predicts the whole forecast horizon in one forward pass rather
than autoregressively. This module uses standard full scaled-dot-product
attention in place of the paper's ProbSparse query-sampling — a
computational optimization (an approximation of full attention) rather than
a different learned function, so this only changes speed characteristics.

## Autoformer

A decomposition-based Transformer forecaster (Wu et al. 2021, NeurIPS).
Series-decomposition blocks (moving-average trend/seasonal split) are
embedded throughout the encoder and decoder rather than applied once up
front. Self-attention is replaced by an Auto-Correlation mechanism that
discovers period-based dependencies via FFT. This module computes a
mathematically-equivalent-in-spirit full (non-top-k) softmax-weighted
circular convolution in place of the official mechanism's dynamic top-k lag
selection + circular shift, which don't translate cleanly to a static,
backend-agnostic Keras graph — see [Implementation Notes](../notes.md).
