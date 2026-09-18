"""
Example mapping: a common PyTorch UNet reference implementation (e.g. the
widely-used `milesial/Pytorch-UNet` naming scheme:
  inc.double_conv.0.weight, down1.maxpool_conv.1.double_conv.0.weight, ...)
onto `keras_climate.remote_sensing.unet.UNet`'s layer names.

`UNet(depth=4)` builds 5 `DoubleConv` stages total: encoder stages
enc0..enc{depth-1} (each followed by a 2x2 maxpool) plus one further
`DoubleConv` at the bottleneck - this last one *is* the "down{depth}" stage
in the milesial naming (it is architecturally just another
maxpool+double_conv step, simply not followed by another downsample). The
decoder then runs `depth` up-stages in *reverse* filter order
(upconv{depth-1}/dec{depth-1} first, consuming the deepest skip
connection, down to upconv0/dec0 last) - this is the reverse order from
milesial's up1..up{depth} numbering (up1 is the *first* decoder block,
consuming the bottleneck + the deepest skip), so the mapping below inverts
the index: `up{k}` -> `dec{depth-k}` / `upconv{depth-k}`.

Each Keras `DoubleConv`/`ConvBNAct` names its inner sublayers "conv"/"bn",
so a conv kernel's full weight path looks like "enc0/conv1/conv/kernel"
and its batchnorm gamma looks like "enc0/conv1/bn/gamma" (there is no
top-level "unet/" prefix - `model.name` is not part of `weight.path`).

This is provided as a worked example of the mapping-rule pattern - adapt
the regexes to whatever specific checkpoint's key naming you're porting
from (state_dict key names vary between UNet reimplementations far more
than for, say, a standard ViT), and adjust `depth` if you built the Keras
model with a non-default `depth`.
"""

import re


def _double_conv_rules(torch_prefix, keras_prefix):
    """(regex, replacement) rules mapping one `*.double_conv.{0,1,3,4}.*`
    PyTorch block (Conv-BN-ReLU-Conv-BN-ReLU, i.e. indices 0=conv1,1=bn1,
    3=conv2,4=bn2 in the `nn.Sequential`) onto one Keras `DoubleConv`."""
    tp = re.escape(torch_prefix)
    return [
        (rf"^{tp}\.double_conv\.0\.weight$", f"{keras_prefix}/conv1/conv/kernel"),
        (rf"^{tp}\.double_conv\.1\.weight$", f"{keras_prefix}/conv1/bn/gamma"),
        (rf"^{tp}\.double_conv\.1\.bias$", f"{keras_prefix}/conv1/bn/beta"),
        (rf"^{tp}\.double_conv\.1\.running_mean$", f"{keras_prefix}/conv1/bn/moving_mean"),
        (rf"^{tp}\.double_conv\.1\.running_var$", f"{keras_prefix}/conv1/bn/moving_variance"),
        (rf"^{tp}\.double_conv\.3\.weight$", f"{keras_prefix}/conv2/conv/kernel"),
        (rf"^{tp}\.double_conv\.4\.weight$", f"{keras_prefix}/conv2/bn/gamma"),
        (rf"^{tp}\.double_conv\.4\.bias$", f"{keras_prefix}/conv2/bn/beta"),
        (rf"^{tp}\.double_conv\.4\.running_mean$", f"{keras_prefix}/conv2/bn/moving_mean"),
        (rf"^{tp}\.double_conv\.4\.running_var$", f"{keras_prefix}/conv2/bn/moving_variance"),
    ]


def build_mapper(depth=4):
    """Returns a `torch_key -> keras_key | None` callable for a
    `UNet(depth=depth)` model, following milesial/Pytorch-UNet naming."""
    rules = []
    rules += _double_conv_rules("inc", "enc0")
    for i in range(1, depth):
        rules += _double_conv_rules(f"down{i}.maxpool_conv.1", f"enc{i}")
    # down{depth} has no further downsample after it - it's the bottleneck.
    rules += _double_conv_rules(f"down{depth}.maxpool_conv.1", "bottleneck")

    for k in range(1, depth + 1):
        dec_idx = depth - k
        rules += [
            (rf"^up{k}\.up\.weight$", f"upconv{dec_idx}/kernel"),
            (rf"^up{k}\.up\.bias$", f"upconv{dec_idx}/bias"),
        ]
        rules += _double_conv_rules(f"up{k}.conv", f"dec{dec_idx}")

    rules += [
        (r"^outc\.conv\.weight$", "logits/kernel"),
        (r"^outc\.conv\.bias$", "logits/bias"),
    ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
