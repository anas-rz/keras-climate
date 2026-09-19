# Weather

`keras_climate.weather` — spatiotemporal nowcasting and weather models.

## ConvLSTM

Stacked ConvLSTM2D encoder-forecaster (Shi et al. 2015) for spatiotemporal
nowcasting (radar reflectivity, precipitation, cloud fields), built on
Keras's built-in `ConvLSTM2D` cells. The decoder unrolls `pred_steps`
future frames autoregressively, each predicted frame fed back as the next
step's input through the *same* decoder layers reused across every step —
a recurrent decoder is only recurrent if its weights are shared across
time.

## Earthformer

A hierarchical spatiotemporal transformer (Gao et al. 2022) built from
"Cuboid Attention" — self-attention restricted to local (T × H × W) cuboids
of the input tensor, with different cuboid shapes/strides per layer to
capture both local and global structure cheaply. Encoder-decoder with a
UNet-like hierarchy (downsample in space between encoder stages, upsample +
skip connections in the decoder).

The official checkpoints are no longer publicly reachable, so this module
is validated against a from-scratch PyTorch reference of its own
(simplified) architecture.

## MetNet / MetNet-2

A convolutional context-aggregating encoder (Sønderby et al. 2020;
Espeholt et al. 2022) — a large receptive field via dilated/strided convs —
followed by temporal encoding across input frames and axial self-attention
across space (row-wise then column-wise, far cheaper than full spatial
self-attention on high-resolution radar fields), producing per-lead-time
precipitation probability maps.

No public code or weights exist for the original MetNet; community
reimplementations use substantially different building blocks that don't
line up with this module's simplified design, so it's validated against a
from-scratch reference of its own architecture — see
[Implementation Notes](../notes.md).

## FourCastNet

A ViT-style global weather forecaster (Pathak et al. 2022, NVIDIA) using
AFNO (see [Neural Operators](operators.md)) as its token-mixer instead of
self-attention, giving O(N log N) spatial mixing suited to the large token
grids a 0.25-degree-resolution ERA5 forecast needs (720×1440 pixels).
Trained to predict the next 6-hour ERA5 atmospheric state from the current
one (autoregressive rollout at inference for multi-step forecasts).

Real pretrained checkpoint: `fourcastnet_backbone` — NVIDIA's official
release, mirrored non-interactively at NERSC. Only
`hard_thresholding_fraction=1.0` is supported, matching the released
checkpoint's own config.

FourCastNet v2 (SFNO) is deliberately not implemented: it replaces AFNO's
planar FFT with a genuine Spherical Harmonic Transform on the sphere, which
would require reproducing a correctness-critical numerical transform
library (`torch-harmonics`) from scratch.

## ClimaX

A ViT-based foundation model for weather/climate (Nguyen et al. 2023,
Microsoft), trained across heterogeneous datasets (CMIP6, ERA5) with
varying variable sets and resolutions. Each input variable/channel gets its
own patch embedding (a separate Conv2D per variable), a learned
per-variable embedding distinguishes them, then a single-query
cross-attention ("variable aggregation") pools the per-variable token
sequences into one unified sequence before a standard ViT encoder; a
lead-time embedding lets one model serve multiple forecast horizons.

Real pretrained checkpoint: `climax_1_40625deg` — Microsoft's official
release, hosted directly on HuggingFace.

## Pangu-Weather

A 3D Earth-Specific Transformer (3DEST) for global medium-range weather
forecasting (Bi et al. 2023, Huawei, *Nature*). Patch-embeds a stack of
13-pressure-level atmospheric variables plus 4 surface variables into a
shared 3D token grid (vertical × lat × lon), processed through a 4-stage
U-Net-style encoder-decoder of windowed 3D transformer blocks with a skip
connection between the first and last stage.

The defining architectural difference from a standard Video-Swin
Transformer is the "Earth-Specific Positional Bias" (ESPB): Earth's
atmosphere is not translation-invariant in latitude or pressure level, so
each (pressure-level-window, latitude-window) pair gets its own full,
densely-parameterized attention bias table. Longitude *is* treated as
translation-invariant (Earth is rotationally symmetric west-to-east), so
the bias table has no longitude-window axis at all. See
[Implementation Notes](../notes.md) for further architectural detail and
input-preprocessing scope boundaries.

Real pretrained checkpoint: `pangu_weather_24` — a clean PyTorch conversion
of Huawei's official ONNX release. **BY-NC-SA 4.0 — non-commercial use
only.** Building the model works on any backend, but a full-resolution
forward pass (721×1440×13-level grid) is memory-heavy enough that it
reliably completes only under the PyTorch backend
(`KERAS_BACKEND=torch`, with `torch.no_grad()`).
