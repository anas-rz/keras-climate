import numpy as np
import pytest
import keras

from keras_climate.forecasting.timesnet import TimesNet, FFTPeriodBlock
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_timesnet_mapper


def test_builds_and_runs():
    model = TimesNet(
        seq_len=48,
        pred_len=24,
        num_channels=3,
        d_model=16,
        d_ff=32,
        num_layers=2,
        num_kernels=3,
        top_k=3,
    )
    x = np.random.randn(2, 48, 3).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 24, 3)


def test_revin_weights_are_tracked():
    model = TimesNet(
        seq_len=24,
        pred_len=8,
        num_channels=2,
        d_model=8,
        d_ff=16,
        num_layers=1,
        num_kernels=2,
        top_k=2,
    )
    assert model.get_layer("revin") in model.layers
    assert {w.path for w in model.weights if "revin" in w.path} == {
        "revin/gamma",
        "revin/beta",
    }


def test_fusion_weights_are_data_dependent():
    block = FFTPeriodBlock(d_model=8, d_ff=16, num_kernels=2, top_k=3)
    x1 = np.random.randn(1, 24, 8).astype("float32") * 0.1
    x2 = np.sin(np.linspace(0, 8 * np.pi, 24))[None, :, None].astype(
        "float32"
    ) * np.ones((1, 24, 8), "float32")
    out1 = keras.ops.convert_to_numpy(block(x1, seq_len=24))
    out2 = keras.ops.convert_to_numpy(block(x2, seq_len=24))
    assert not np.allclose(out1 - x1, out2 - x2, atol=1e-3)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class InceptionBlockV1(nn.Module):
        def __init__(self, in_ch, out_ch, num_kernels):
            super().__init__()
            self.convs = nn.ModuleList(
                [
                    nn.Conv2d(in_ch, out_ch, 2 * i + 1, padding=i)
                    for i in range(num_kernels)
                ]
            )

        def forward(self, x):
            return torch.stack([c(x) for c in self.convs], dim=-1).mean(dim=-1)

    class TorchRevIN(nn.Module):
        def __init__(self, num_features, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        def norm(self, x):
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = (
                (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            )
            return (x - self.mean) / self.stdev * self.affine_weight + self.affine_bias

        def denorm(self, x):
            x = (x - self.affine_bias) / (self.affine_weight + self.eps**2)
            return x * self.stdev + self.mean

    def candidate_periods(seq_len, k=5):
        base = [24, 12, 8, 6, 4]
        return [p for p in base[:k] if p < seq_len] or [2]

    class TimesBlock(nn.Module):
        def __init__(self, d_model, d_ff, num_kernels, top_k):
            super().__init__()
            self.conv1 = InceptionBlockV1(d_model, d_ff, num_kernels)
            self.conv2 = InceptionBlockV1(d_ff, d_model, num_kernels)
            self.top_k = top_k

        def forward(self, x, seq_len):
            B, L, D = x.shape
            periods = candidate_periods(seq_len, self.top_k)
            x_time_last = x.transpose(1, 2)
            xf = torch.fft.rfft(x_time_last, dim=-1)
            amplitude = xf.abs().mean(dim=1)

            results, weights = [], []
            for period in periods:
                pad_len = ((seq_len + period - 1) // period) * period
                if pad_len != seq_len:
                    x_pad = F.pad(x, (0, 0, 0, pad_len - seq_len))
                else:
                    x_pad = x
                num_periods = pad_len // period
                reshaped = x_pad.reshape(B, num_periods, period, D).permute(0, 3, 1, 2)
                out = F.gelu(self.conv1(reshaped))
                out = self.conv2(out)
                out = out.permute(0, 2, 3, 1).reshape(B, pad_len, D)[:, :seq_len, :]
                results.append(out)
                freq_idx = min(seq_len // period, seq_len // 2)
                weights.append(amplitude[:, freq_idx])

            stacked = torch.stack(results, dim=-1)
            w = torch.stack(weights, dim=-1).softmax(dim=-1)[:, None, None, :]
            return (stacked * w).sum(dim=-1) + x

    class TorchTimesNet(nn.Module):
        def __init__(
            self,
            seq_len,
            pred_len,
            num_channels,
            d_model,
            d_ff,
            num_layers,
            num_kernels,
            top_k,
        ):
            super().__init__()
            self.revin = TorchRevIN(num_channels)
            self.value_embed = nn.Linear(num_channels, d_model)
            self.timesblocks = nn.ModuleList(
                [
                    TimesBlock(d_model, d_ff, num_kernels, top_k)
                    for _ in range(num_layers)
                ]
            )
            self.norms = nn.ModuleList(
                [nn.LayerNorm(d_model, eps=1e-6) for _ in range(num_layers)]
            )
            self.predict_linear = nn.Linear(seq_len, seq_len + pred_len)
            self.output_proj = nn.Linear(d_model, num_channels)
            self.seq_len, self.pred_len = seq_len, pred_len

        def forward(self, x):
            x = self.revin.norm(x)
            x = self.value_embed(x)
            for blk, norm in zip(self.timesblocks, self.norms):
                x = norm(blk(x, self.seq_len))
            x = self.predict_linear(x.transpose(1, 2)).transpose(1, 2)
            out = self.output_proj(x)[:, -self.pred_len :, :]
            return self.revin.denorm(out)

    torch.manual_seed(0)
    seq_len, pred_len, num_channels = 24, 8, 2
    d_model, d_ff, num_layers, num_kernels, top_k = 8, 16, 1, 2, 2

    torch_model = TorchTimesNet(
        seq_len, pred_len, num_channels, d_model, d_ff, num_layers, num_kernels, top_k
    )
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    flat = {}
    flat["revin.affine_weight"] = torch_model.revin.affine_weight.detach().numpy()
    flat["revin.affine_bias"] = torch_model.revin.affine_bias.detach().numpy()
    flat["value_embed.weight"] = torch_model.value_embed.weight.detach().numpy()
    flat["value_embed.bias"] = torch_model.value_embed.bias.detach().numpy()
    for i, (blk, norm) in enumerate(zip(torch_model.timesblocks, torch_model.norms)):
        for conv_name in ("conv1", "conv2"):
            for k, conv in enumerate(getattr(blk, conv_name).convs):
                flat[f"timesblock{i}.{conv_name}.convs.{k}.weight"] = (
                    conv.weight.detach().numpy()
                )
                flat[f"timesblock{i}.{conv_name}.convs.{k}.bias"] = (
                    conv.bias.detach().numpy()
                )
        flat[f"norm{i}.weight"] = norm.weight.detach().numpy()
        flat[f"norm{i}.bias"] = norm.bias.detach().numpy()
    flat["predict_linear.weight"] = torch_model.predict_linear.weight.detach().numpy()
    flat["predict_linear.bias"] = torch_model.predict_linear.bias.detach().numpy()
    flat["output_proj.weight"] = torch_model.output_proj.weight.detach().numpy()
    flat["output_proj.bias"] = torch_model.output_proj.bias.detach().numpy()

    keras_model = TimesNet(
        seq_len=seq_len,
        pred_len=pred_len,
        num_channels=num_channels,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_kernels=num_kernels,
        top_k=top_k,
    )
    mapper = build_timesnet_mapper(num_layers=num_layers, num_kernels=num_kernels)
    report = WeightConverter(keras_model, flat, mapper).convert(
        strict=True, verbose=False
    )
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(2, seq_len, num_channels).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model(x_np, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert (
        max_diff < 1e-2
    ), f"TimesNet weight port numerical mismatch: max abs diff {max_diff}"
