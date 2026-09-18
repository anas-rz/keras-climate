"""Build/shape sanity checks + PyTorch weight-port round-trip test for
`keras_climate.remote_sensing.unet.UNet`.

Run with: pytest keras_climate/remote_sensing/test_unet.py
"""
import numpy as np
import pytest
import keras

from .unet import UNet, unet_config
from ..weights import WeightConverter
from ..weights.mappings import build_unet_mapper
from ..weights.pretrained import unet_carvana


# --------------------------------------------------------------------------
# Build / forward-pass sanity checks (Keras only, no torch required)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["small", "base", "large"])
def test_builds_and_runs(variant):
    cfg = unet_config(variant)
    model = UNet(input_shape=(64, 64, 4), num_classes=5, **cfg)
    x = np.random.randn(2, 64, 64, 4).astype("float32")
    y = model(x)
    assert tuple(y.shape) == (2, 64, 64, 5)


def test_custom_depth_and_odd_number_of_classes():
    model = UNet(input_shape=(64, 64, 3), num_classes=1, depth=3, base_filters=16)
    x = np.random.randn(1, 64, 64, 3).astype("float32")
    y = model(x)
    assert tuple(y.shape) == (1, 64, 64, 1)


def test_final_activation_applied():
    model = UNet(input_shape=(32, 32, 3), num_classes=1, depth=2, base_filters=8,
                 final_activation="sigmoid")
    x = np.random.randn(1, 32, 32, 3).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.min() >= 0.0 and y.max() <= 1.0


# --------------------------------------------------------------------------
# PyTorch weight-port round-trip: a reference UNet matching
# milesial/Pytorch-UNet's naming (the convention `build_unet_mapper`
# assumes) is built, its (random) weights are ported through the real
# converter + mapping used in this repo, and the two forward passes are
# compared numerically end-to-end.
# --------------------------------------------------------------------------

def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class OutConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    class TorchUNet(nn.Module):
        """depth=4, base_filters=64 - matches `UNet()`'s defaults."""

        def __init__(self, n_channels=3, n_classes=2, base=64):
            super().__init__()
            self.inc = DoubleConv(n_channels, base)
            self.down1 = Down(base, base * 2)
            self.down2 = Down(base * 2, base * 4)
            self.down3 = Down(base * 4, base * 8)
            self.down4 = Down(base * 8, base * 16)
            self.up1 = Up(base * 16, base * 8)
            self.up2 = Up(base * 8, base * 4)
            self.up3 = Up(base * 4, base * 2)
            self.up4 = Up(base * 2, base)
            self.outc = OutConv(base, n_classes)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            return self.outc(x)

    torch.manual_seed(0)
    torch_model = TorchUNet(n_channels=3, n_classes=2, base=64)
    torch_model.eval()
    # Randomize BatchNorm into a non-trivial eval state so a wrong BN
    # mapping/epsilon would actually be visible numerically.
    with torch.no_grad():
        for m in torch_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_(0.0, 0.1)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()
                  if "num_batches_tracked" not in k}

    keras_model = UNet(input_shape=(64, 64, 3), num_classes=2, base_filters=64, depth=4)
    keras_model(np.zeros((1, 64, 64, 3), dtype="float32"))  # build

    mapper = build_unet_mapper(depth=4)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, 3, 64, 64).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()  # (B, C, H, W)

    keras_in = np.transpose(x_np, (0, 2, 3, 1))  # NCHW -> NHWC
    keras_out = keras.ops.convert_to_numpy(keras_model(keras_in, training=False))
    keras_out = np.transpose(keras_out, (0, 3, 1, 2))  # NHWC -> NCHW

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-3, f"UNet weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_carvana_checkpoint():
    """Downloads milesial/Pytorch-UNet's real released Carvana checkpoint
    and confirms it loads cleanly and produces a sane binary mask.
    Run explicitly with `pytest -m pretrained` (not part of the default
    run - this hits the network and downloads a ~124MB file, cached under
    `KERAS_CLIMATE_CACHE_DIR` / `~/.cache/keras_climate/weights` after the
    first run)."""
    pytest.importorskip("torch")
    model, report = unet_carvana(input_shape=(256, 256, 3))
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x = np.random.rand(1, 256, 256, 3).astype("float32")
    logits = keras.ops.convert_to_numpy(model(x, training=False))
    assert logits.shape == (1, 256, 256, 2)
    assert np.isfinite(logits).all()
