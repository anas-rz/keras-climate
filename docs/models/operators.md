# Neural Operators

`keras_climate.operators` — neural operator learning (PDE surrogate
modeling). Unlike a standard CNN, these are generally resolution-invariant:
the same learned weights apply regardless of the input grid's spatial size.

## AFNO

Adaptive Fourier Neural Operator (Guibas et al. 2021) — a token-mixing
block for ViT-style backbones that replaces self-attention with a 2D FFT
over the spatial token grid, a block-diagonal complex-valued 2-layer MLP
applied per retained frequency mode, soft-thresholding sparsification, and
an inverse FFT — all at O(N log N) instead of self-attention's O(N²). This
is the token-mixer used inside [FourCastNet](weather.md#fourcastnet). Only
`hard_thresholding_fraction=1.0` (every frequency mode kept) is supported —
the setting FourCastNet's own released checkpoint actually uses.

No standalone, task-agnostic pretrained AFNO checkpoint exists anywhere
(the only released AFNO weights are FourCastNet's own) — see
[Pretrained Weights](../pretrained-weights.md).

## FNO

Fourier Neural Operator (Li et al. 2021) — a lifting layer raises the input
to a hidden "width", a stack of Fourier layers each apply a 2D FFT,
truncate to the lowest `modes1 × modes2` frequencies, multiply by a learned
complex weight tensor per retained mode, zero-pad back to the full spectrum
and inverse-FFT, in parallel with a pointwise (1×1) conv "skip" path — and
a projection head maps back to the output channel count.

A real PDEBench-trained checkpoint exists on HuggingFace, but its
state_dict reveals a modified block structure that doesn't match the
original paper's code, PDEBench's own repo, or any indexed version of the
`neuraloperator` library — see [Pretrained Weights](../pretrained-weights.md)
for why it isn't ported here.

## DeepONet

Learns a mapping from an input *function* (sampled at a fixed set of
sensor points, encoded by a "branch net" MLP) to an output function
evaluated at an arbitrary query coordinate (encoded by a "trunk net" MLP)
(Lu et al. 2021) — the two encodings are combined via a dot product plus a
learned scalar bias. Unlike a grid-based operator (FNO/UNO/AFNO),
DeepONet's output can be queried at any continuous coordinate, not just
points on the input's discretization grid.

A real checkpoint exists, but wraps the branch net with a custom
multi-scale CNN feature extractor whose exact wiring can't be determined
from the checkpoint alone — see [Pretrained Weights](../pretrained-weights.md).

## UNO

U-shaped Neural Operator (Rahman et al. 2022) — a U-Net-style
encoder-decoder where each conv block is a spectral (FNO-style) operator
instead of a local convolution, so the whole network stays
resolution-invariant while still getting U-Net's multi-scale
skip-connection structure. This module uses standard pooling/upsampling
between levels for resolution changes (rather than changing resolution
directly via the FFT, as in the original paper), keeping each level's
spectral conv at a fixed resolution — a simpler, easier-to-verify structure
without the added complexity of a resolution-changing spectral transform.

No pretrained checkpoint exists for UNO anywhere.
