# Neural Operators

`keras_climate.operators` — neural operator learning (PDE surrogate
modeling), scoped to the one operator architecture with a direct climate
application in this repo. Unlike a standard CNN, it is resolution-invariant:
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

!!! note
    Generic, climate-agnostic operator architectures (FNO, DeepONet, UNO —
    typically trained/benchmarked on fluid-dynamics or Darcy-flow PDEs, not
    Earth-system data) have been removed from this repo's scope. AFNO stays
    because it's the actual building block a real climate model
    ([FourCastNet](weather.md#fourcastnet)) uses.
