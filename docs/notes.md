# Implementation Notes

Non-obvious architectural decisions, quirks, and Keras-specific gotchas
that are easy to "fix" incorrectly if you don't know why they're there.

## AnySat release port (`foundation/anysat_release.py`)

**iRPE head-0 broadcasting quirk.** The released config only turns on the
contextual relative-position-encoding bias *on keys* (`rpe_on="k"`), shared
across heads, with 8 relative-distance buckets. A non-obvious,
empirically-confirmed quirk of the reference implementation's pure-Python
(non-CUDA) fallback path is that `torch.gather`'s under-specified-dim
broadcasting silently restricts the bias computation to *only the first
attention head's queries* — the other heads' queries never participate —
with the resulting bias then broadcast back across all heads when added to
the attention scores. This is reproduced bit-for-bit in `_rpe_bias_k` since
the released checkpoint's weights were trained against exactly that
behavior; "fixing" it would silently break loading the real weights, not
correct a bug.

**LayerNorm eps.** `AnyModule`'s own blocks use `eps=1e-6`, but the global
and cross-pooling blocks use PyTorch's `nn.LayerNorm` default of `eps=1e-5`
(Keras's own default is `1e-3`) — the split must be reproduced exactly, not
unified to one value, to match the checkpoint numerically.

Every reshape/grouping op (the nested "unfold" sequences that interleave a
modality's finer patch grid with the coarser `scale`-tile grid) was
validated index-for-index against the actual `gastruc/AnySat` source, not
re-derived from the paper.

## Clay and CROMA: LayerNorm eps

Both `foundation/clay.py` and `foundation/croma.py` hardcode
`_LN_EPS = 1e-5`: PyTorch's plain `nn.LayerNorm`/`nn.TransformerEncoderLayer`
default to `eps=1e-5`, while Keras's own default is `1e-3` and this repo's
*other* ViT blocks use `1e-6` (matching timm/MAE). Each model's eps must
match its own reference implementation's default exactly for checkpoint
compatibility — there is no single "correct" value across the repo.

## Pangu-Weather (`weather/pangu_weather.py`)

**Earth-Specific Positional Bias.** Earth's atmosphere is not
translation-invariant in latitude or pressure level (polar vs. equatorial
dynamics genuinely differ, as does behavior at different altitudes), so the
attention bias is *not* shared/relative across window positions in those
two axes — each (pressure-level-window, latitude-window) pair gets its own
full, densely-parameterized bias table. Longitude *is* treated as
translation-invariant, so the bias table has no longitude-window axis.

**Batch size.** The official reference hardcodes several reshapes assuming
`batch_size=1` (the realistic use case for a single 721×1440 global weather
state). This port generalizes those reshapes to carry a real batch axis
(merging it into the longitude-window axis, which the bias never depends
on) so `batch_size > 1` also works, while remaining bit-exact with the
reference for `batch=1`.

**`attn_mask`, not `mask`.** `EarthAttention3D.call`'s optional mask
argument is deliberately named `attn_mask`, not `mask` — Keras's
`Layer.__call__` reserves the literal name `mask` for its own automatic
mask-propagation protocol. A custom `call()` kwarg named `mask` can be
silently intercepted by the framework instead of reaching your own code as
a plain value.

**Naming mirrors the reference exactly**, including unusual choices (the
MLP submodule is called `linear`, its own two Dense layers `linear1`/
`linear2`; attention QKV/output projections are also `linear1`/`linear2`
rather than `qkv`/`proj`) — kept as-is (rather than "improved") specifically
so `weights/mappings/pangu_weather_mapping.py`'s regex rules stay a direct
one-to-one translation of the real checkpoint's own key names. The shifted
window mask generator (`_gen_shift_mask`) similarly keeps a latitude-axis
slicing asymmetry byte-for-byte faithful to the reference, whether or not
it was intentional upstream.

**Scope boundary.** `PatchEmbedding` expects its auxiliary inputs
(per-variable normalization stats, land-sea/soil-type/topography masks, a
constant per-level field) already normalized and concatenated by the
caller — the official `aux_data.zip` preprocessing is not baked into the
model itself.

## PatchTST and TFT: the Functional-graph "dead branch" gotcha

A layer whose weights are only ever touched outside of `call()` never
becomes a node in the Functional graph, so Keras has no way to know it
belongs to the model — those weights would then silently be missing from
`model.weights` entirely, meaning `model.fit()` never updates them and
`save_weights()` silently drops them. This showed up twice:

- `PatchTST`'s `RevIN` invokes `normalize`/`denormalize` through the real
  `call()`/`__call__()` path (`revin(x, mode="norm")`), not via custom
  methods that bypass it.
- `PatchTST`'s `LearnedPositionalEmbedding` adds its raw weight directly
  inside `call()` on the real (symbolic) token tensor, rather than doing an
  `Embedding` lookup over an eagerly-computed constant index array (which
  would never register as a graph node).
- `TFT` deliberately does *not* implement the paper's extra
  `static_context_selection` GRN input, since there's no point constructing
  a GRN whose output would never be consumed — an unconsumed branch's
  weights wouldn't register as part of the model either.

## TimesNet: a fixed fusion bug

The per-period fusion weights must actually depend on the input's FFT
amplitude, not be a uniform average. An earlier version had a bug where
`xf`/`amp` were computed and immediately discarded, so every period got the
same fusion weight regardless of input — covered by a regression test.

## `weights/converter.py`: checkpoint batch-broadcast axis

A very common real-world checkpoint quirk: a positional/token embedding
saved with an extra leading batch-broadcast axis of size 1 (e.g. official
MAE's `pos_embed` is `(1, N, D)`) where this repo's Keras weight is stored
without it (`(N, D)`) and adds the axis back at use time. `convert()`
squeezes it before shape inference so it isn't treated as a hard mismatch.

Also: an explicit `param_kind` override is always honored *before* the
"shapes already match" shortcut in `infer_transpose` — a square
`nn.Linear.weight` (`out == in`) has the exact same shape as its Keras
`Dense.kernel` counterpart while still needing the `(out, in) -> (in, out)`
transpose, so shape equality alone can't be trusted to mean "no transform
needed" for a known dense kernel.

## `utils/layers.py`: `"same"` padding vs. PyTorch's symmetric padding

Keras's `padding="same"` pads *asymmetrically* (extra on the bottom/right)
whenever `strides > 1` and the input size isn't an exact multiple of the
stride in a way that makes the needed padding even. PyTorch reference convs
almost always pad symmetrically via an explicit
`padding=dilation*(k-1)//2`, so relying on `"same"` would silently misalign
every ported strided conv (stem/downsampling convs) against its source
checkpoint. `ConvBNAct` pads explicitly + `"valid"` whenever `strides > 1`
to reproduce PyTorch's convention exactly; for `strides == 1`, `"same"` is
already symmetric.

## SatCLIP: reverse-engineered spherical harmonics normalization

The official checkpoint's default `harmonics_calculation="analytic"`
sign/normalization convention (a ~40,000-line sympy-generated closed-form
expansion upstream, one function per `(l, m)` pair) is not reproduced
directly — it was reverse-engineered from its output values instead:
standard 4π-normalized real spherical harmonics without the
Condon-Shortley phase, *except* the `m=0` (zonal) terms carry an extra
factor of π that the official "analytic" functions apply but the
"closed-form" alternative in the same codebase does not. Confirmed against
known reference outputs (`Yl0_m0 = 0.886226925...`, `Yl1_m0 =
1.534990062...`, both exactly π times the standard-normalized value).

## Models with no ported or verifiable pretrained checkpoint

Several modules include a scope note on why a real, publicly available
checkpoint was *not* ported even though one technically exists — the
underlying issue is always the same shape: the checkpoint's state_dict
reveals a computation graph that can't be reconstructed from tensor
names/shapes alone without the original source code, so porting it would
risk silently assigning real weights into the wrong graph. See
[Pretrained Weights](pretrained-weights.md#no-loader-available) for the
full list (FNO, DeepONet, RingMo, Clay, this repo's own `AnySatEncoder`,
Autoformer, Informer, N-BEATS, DLinear, Earthformer, MetNet).
