# Foundation Models

`keras_climate.foundation` — large pretrained Earth-observation foundation
models.

## Prithvi

A ViT Masked Autoencoder (IBM/NASA, 2023) pretrained on NASA Harmonized
Landsat-Sentinel (HLS) imagery, using 3D (tubelet) patch embedding over a
short temporal stack (typically 3 timesteps) of 6-band multispectral
imagery. Downstream tasks (e.g. burn-scar or flood mapping) attach a
segmentation/classification head to `PrithviEncoder`.

Prithvi-EO-2.0 (the 2024 successor) reuses this exact same `PrithviEncoder`
architecture unchanged — just at different
embed_dim/depth/num_heads/patch_size/num_frames — so no new model code is
needed for its base (non-"-TL") checkpoint variants. The "-TL"
(Temporal+Location) variants additionally add a `TemporalEncoder` /
`LocationEncoder` this module does not yet implement.

Real pretrained checkpoints: `prithvi_eo_100m` (Prithvi-EO-1.0-100M) and
`prithvi_eo_v2_300m` (Prithvi-EO-2.0-300M), both IBM/NASA's official
releases, encoder only.

## Clay

A ViT-MAE-style foundation model (Clay Foundation, 2024) that generalizes
across sensors via a DOFA-style *dynamic* patch embedding: instead of a
fixed per-sensor conv kernel, a small hypernetwork generates the
patch-embedding conv weights on the fly, conditioned on each band's
wavelength — so a new sensor with a different band count/order than
anything seen in training can still be embedded sensibly, without
retraining or a fixed input-channel contract.

This is a faithful port of the official implementation, matching three
easy-to-get-wrong details in particular: the patch embedding is
*generated* (not learned directly) by a mini `nn.TransformerEncoder` over
per-band wavelength sincos features; position encoding is a GSD-scaled 2D
sincos embedding with 8 channels reserved for pre-encoded time/lat-lon
metadata, concatenated rather than added; and the main transformer
backbone follows lucidrains' vit-pytorch style (bias-free `to_qkv`/`to_out`,
LayerNorm inside the FFN's `Sequential`), a different convention from this
repo's other ViT blocks (which follow timm's convention).

No public Clay checkpoint is small enough to validate here (the official
v1.5 release is ~5GB), so this module is validated against a from-scratch
PyTorch reference instead — treat the architecture as faithfully ported,
but budget for possible checkpoint-specific naming adjustments.

## CROMA

A dual-encoder foundation model (Fuller et al. 2023) that jointly
represents SAR (e.g. Sentinel-1) and optical (e.g. Sentinel-2) imagery.
Each modality has its own ViT encoder; a cross-attention fusion stack
produces a joint multimodal representation. Pretraining combines a
contrastive objective (aligning the two modalities' global representations)
with per-modality masked-image modeling.

This is a faithful port of the official implementation, matching three
architectural choices that are easy to get wrong by "reasonable-sounding"
default assumptions: patch embedding is a plain `Linear` over flattened raw
pixel patches, not a Conv2D; there is no learned position embedding at all
— every attention layer instead adds a fixed (non-trainable) 2D ALiBi bias
computed once from pairwise patch-grid distances; and the SAR encoder is
intentionally *half* the depth of the optical encoder, with the joint/cross
encoder a single query stream (SAR) progressively attending against a
*fixed* optical context, not a bidirectional update of both streams.

Real pretrained checkpoints: `croma_base` and `croma_large`, the paper
authors' own released weights, both modalities + the joint fusion encoder.

## AnySat

`keras_climate.foundation` ships **two, deliberately different**
implementations of AnySat (Astruc et al. 2024):

- **`AnySatEncoder`** (`anysat.py`) — this repo's own simpler "any
  modality, any resolution" ViT: every modality's patches are projected
  into a shared embedding space and tagged with a learned modality
  embedding plus a resolution-aware position embedding (patch center
  coordinates scaled by ground-sample distance), so tokens from different
  resolutions/modalities are spatially comparable. This lets one model
  jointly consume, e.g., 10m Sentinel-2, 20m Sentinel-1, and 30m Landsat
  over the same footprint. Validated only against a from-scratch synthetic
  reference — **not** weight-compatible with the real public checkpoint.

- **`AnySatRelease`** (`anysat_release.py`) — a faithful port of the
  *officially released* architecture, loadable with the real public
  checkpoint (`g-astruc/AnySat` on HuggingFace, "base" size: embed_dim=768,
  depth=6, num_heads=12, 125.9M params). It has per-modality-kind
  projectors (image vs. time-series), a shared local encoder, a global
  encoder, and a final cross-attention pooling block. See
  [Implementation Notes](../notes.md) for the iRPE head-0 broadcasting
  quirk this port reproduces bit-for-bit to stay weight-compatible.

Real pretrained checkpoint: `anysat_base`, backed by `AnySatRelease`,
validated end-to-end to ~1e-6 against real PyTorch forward passes using the
actual checkpoint weights.
