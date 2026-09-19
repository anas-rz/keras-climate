import os
import numpy as np
import pytest
import keras

from keras_climate.weather.pangu_weather import (
    PanguWeather, EarthAttention3D, EarthSpecificBlock,
)
from keras_climate.weights import WeightConverter


def test_builds():
    model = PanguWeather()
    assert len(model.weights) == 231
    assert model.input_shape == [(None, 13, 721, 1440, 6), (None, 721, 1440, 7)]
    assert model.output_shape == [(None, 5, 13, 721, 1440), (None, 4, 721, 1440)]


def test_earth_attention_3d_small_scale():
    layer = EarthAttention3D(dim=8, heads=2, type_of_windows=4)
    x = np.random.randn(3, 4, 2 * 6 * 12, 8).astype("float32")
    out = keras.ops.convert_to_numpy(layer(x))
    assert out.shape == (3, 4, 2 * 6 * 12, 8)


def test_earth_specific_block_shift_and_no_shift():
    block_plain = EarthSpecificBlock(dim=8, heads=2, resolution=(4, 8, 24), shift=False)
    block_shift = EarthSpecificBlock(dim=8, heads=2, resolution=(4, 8, 24), shift=True)
    x = np.random.randn(2, 4 * 8 * 24, 8).astype("float32")
    out_plain = keras.ops.convert_to_numpy(block_plain(x))
    out_shift = keras.ops.convert_to_numpy(block_shift(x))
    assert out_plain.shape == (2, 4 * 8 * 24, 8)
    assert out_shift.shape == (2, 4 * 8 * 24, 8)
    assert not np.allclose(out_plain, out_shift)


def test_weight_port_roundtrip_matches_pytorch_reference_small_scale():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    window_size = (2, 6, 12)

    class TorchMlp(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim * 4)
            self.linear2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.linear2(F.gelu(self.linear1(x)))

    class TorchEarthAttention3D(nn.Module):
        def __init__(self, dim, heads, type_of_windows):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim * 3, bias=True)
            self.linear2 = nn.Linear(dim, dim)
            self.head_number = heads
            self.dim = dim
            self.scale = (dim // heads) ** -0.5
            self.type_of_windows = type_of_windows
            vol = window_size[0] * window_size[1] * window_size[2]
            self.earth_specific_bias = nn.Parameter(torch.zeros(1, type_of_windows, heads, vol, vol))

        def forward(self, x, mask):
            original_shape = x.shape
            x = self.linear1(x)
            qkv = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3, self.head_number, self.dim // self.head_number)
            qkv = qkv.permute(3, 0, 1, 4, 2, 5)
            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            attn = q @ k.transpose(-2, -1)
            attn = attn + self.earth_specific_bias
            if mask is not None:
                nW = mask.shape[0]
                nWB = attn.shape[0]
                attn = attn.view(nWB // nW, nW, self.type_of_windows, self.head_number,
                                 attn.shape[-2], attn.shape[-1]) + mask.unsqueeze(2).unsqueeze(0)
                attn = attn.reshape(nWB, self.type_of_windows, self.head_number,
                                    attn.shape[-2], attn.shape[-1])
            attn = attn.softmax(dim=-1)
            x = attn @ v
            x = x.permute(0, 1, 3, 2, 4).reshape(original_shape)
            return self.linear2(x)

    def gen_mask(Z, H_pad, W, device):
        wZ, wH, wW = window_size
        img_mask = torch.zeros((1, Z, H_pad, W, 1))
        z_slices = (slice(0, -wZ), slice(-wZ, -wZ // 2), slice(-wZ // 2, None))
        h_slices = (slice(0, -wH), slice(wH, -wH // 2), slice(-wH // 2, None))
        cnt = 0
        for z in z_slices:
            for h in h_slices:
                img_mask[:, z, h, :, :] = cnt
                cnt += 1
        type_of_windows = (Z // wZ) * (H_pad // wH)
        img_mask = img_mask.reshape(1, Z // wZ, wZ, H_pad // wH, wH, W // wW, wW, 1)
        img_mask = img_mask.permute(0, 5, 1, 3, 2, 4, 6, 7)
        mask_windows = img_mask.reshape(-1, type_of_windows, wZ * wH * wW)
        attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    class TorchEarthSpecificBlock(nn.Module):
        def __init__(self, dim, heads, resolution, shift):
            super().__init__()
            self.Z, self.H, self.W = resolution
            wZ, wH, wW = window_size
            self.H_pad = self.H + (wH - self.H % wH) % wH
            self.type_of_windows = (self.Z // wZ) * (self.H_pad // wH)
            self.shift = shift
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.linear = TorchMlp(dim)
            self.attention = TorchEarthAttention3D(dim, heads, self.type_of_windows)
            self.padding_front, self.padding_back = 0, self.H_pad - self.H

        def forward(self, x):
            Z, H, W = self.Z, self.H, self.W
            wZ, wH, wW = window_size
            shortcut = x
            x = x.view(x.shape[0], Z, H, W, x.shape[2])
            x = F.pad(x, (0, 0, 0, 0, self.padding_front, self.padding_back), "constant")
            ori_shape = x.shape

            mask = None
            if self.shift:
                x = torch.roll(x, shifts=[-wZ // 2, -wH // 2, -wW // 2], dims=(1, 2, 3))
                mask = gen_mask(Z, self.H_pad, W, x.device)

            x_window = x.view(x.shape[0], Z // wZ, wZ, self.H_pad // wH, wH, W // wW, wW, x.shape[-1])
            x_window = x_window.permute(0, 5, 1, 3, 2, 4, 6, 7)
            x_window = x_window.reshape(x_window.shape[0] * x_window.shape[1], self.type_of_windows,
                                        wZ, wH, wW, x_window.shape[-1])
            x_window = x_window.reshape(x_window.shape[0], x_window.shape[1], wZ * wH * wW, x_window.shape[-1])

            attn = self.attention(x_window, mask)

            x_shifted = attn.view(x.shape[0], W // wW, Z // wZ, self.H_pad // wH, wZ, wH, wW, -1)
            x_shifted = x_shifted.permute(0, 2, 4, 3, 5, 1, 6, 7)
            x_shifted = x_shifted.contiguous().view(ori_shape)

            if self.shift:
                x = torch.roll(x_shifted, shifts=[wZ // 2, wH // 2, wW // 2], dims=(1, 2, 3))
            else:
                x = x_shifted

            x = x[:, :, self.padding_front:x.shape[2] - self.padding_back, :, :]
            x = x.contiguous().view(x.shape[0], Z * H * W, x.shape[-1])

            x = shortcut + self.norm1(x)
            x = x + self.norm2(self.linear(x))
            return x

    torch.manual_seed(0)
    dim, heads, resolution = 8, 2, (4, 8, 24)

    for shift in (False, True):
        torch_block = TorchEarthSpecificBlock(dim, heads, resolution, shift)
        torch_block.eval()
        with torch.no_grad():
            for p in torch_block.parameters():
                p.normal_(0.0, 0.2)

        state_dict = {k: v.detach().numpy() for k, v in torch_block.state_dict().items()}

        keras_block = EarthSpecificBlock(dim, heads, resolution, shift=shift, name="block")
        x0 = np.zeros((1, resolution[0] * resolution[1] * resolution[2], dim), dtype="float32")
        keras_block(x0)

        mapper = {
            "norm1.weight": "block/norm1/gamma", "norm1.bias": "block/norm1/beta",
            "norm2.weight": "block/norm2/gamma", "norm2.bias": "block/norm2/beta",
            "linear.linear1.weight": "block/linear/linear1/kernel",
            "linear.linear1.bias": "block/linear/linear1/bias",
            "linear.linear2.weight": "block/linear/linear2/kernel",
            "linear.linear2.bias": "block/linear/linear2/bias",
            "attention.earth_specific_bias": "block/attention/earth_specific_bias",
            "attention.linear1.weight": "block/attention/linear1/kernel",
            "attention.linear1.bias": "block/attention/linear1/bias",
            "attention.linear2.weight": "block/attention/linear2/kernel",
            "attention.linear2.bias": "block/attention/linear2/bias",
        }
        report = WeightConverter(keras_block, state_dict, lambda k: mapper.get(k)).convert(
            strict=False, verbose=False)
        assert all(k.endswith("attn_mask") for k in report["missing_in_source"])
        assert not report["unused_source_keys"]

        x_np = np.random.randn(2, resolution[0] * resolution[1] * resolution[2], dim).astype("float32")
        with torch.no_grad():
            torch_out = torch_block(torch.from_numpy(x_np)).numpy()
        keras_out = keras.ops.convert_to_numpy(keras_block(x_np, training=False))

        max_diff = np.abs(torch_out - keras_out).max()
        assert max_diff < 1e-2, (
            f"EarthSpecificBlock (shift={shift}) weight port numerical mismatch: max abs diff {max_diff}")


@pytest.mark.pretrained
def test_real_pretrained_pangu_weather_matches_reference_implementation():
    if keras.backend.backend() != "torch":
        pytest.skip("PanguWeather's full-resolution forward pass needs the torch backend - "
                    "see module docstring. Run with KERAS_BACKEND=torch.")
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F
    from collections import OrderedDict
    from keras_climate.weights.mappings import build_pangu_weather_mapper, convert_pangu_weather_state_dict
    from keras_climate.weights.pretrained import pangu_weather_24, DEFAULT_CACHE_DIR

    window_size = (2, 6, 12)

    class Mlp(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim * 4)
            self.linear2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.linear2(F.gelu(self.linear1(x)))

    class EarthAttention3D(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim * 3, bias=True)
            self.linear2 = nn.Linear(dim, dim)
            self.head_number = heads
            self.dim = dim
            self.scale = (dim // heads) ** -0.5
            input_shape = [8, 186] if dim == 192 else [8, 96]
            self.type_of_windows = (input_shape[0] // window_size[0]) * (input_shape[1] // window_size[1])
            vol = window_size[0] * window_size[1] * window_size[2]
            self.earth_specific_bias = nn.Parameter(torch.zeros(1, self.type_of_windows, heads, vol, vol))

        def forward(self, x, mask):
            original_shape = x.shape
            x = self.linear1(x)
            qkv = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 3, self.head_number,
                                    self.dim // self.head_number))
            qkv = torch.permute(qkv, (3, 0, 1, 4, 2, 5))
            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            attn = q @ k.transpose(-2, -1)
            attn = attn + self.earth_specific_bias
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(1, nW, self.type_of_windows, self.head_number,
                                 attn.shape[-2], attn.shape[-1]) + mask.unsqueeze(2).unsqueeze(0)
                attn = attn.reshape(nW, self.type_of_windows, self.head_number,
                                    attn.shape[-2], attn.shape[-1])
            attn = attn.softmax(dim=-1)
            x = attn @ v
            x = torch.permute(x, (0, 1, 3, 2, 4))
            x = torch.reshape(x, original_shape)
            return self.linear2(x)

    class EarthSpecificBlock(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.linear = Mlp(dim)
            self.attention = EarthAttention3D(dim, heads)
            self.padding_front, self.padding_back = 0, 5
            input_shape = [8, 186] if dim == 192 else [8, 96]
            self.type_of_windows = (input_shape[0] // window_size[0]) * (input_shape[1] // window_size[1])

        def gen_mask(self, x):
            wZ, wH, wW = window_size
            img_mask = torch.zeros((1, x.shape[1], x.shape[2], x.shape[3], 1))
            z_slices = (slice(0, -wZ), slice(-wZ, -wZ // 2), slice(-wZ // 2, None))
            h_slices = (slice(0, -wH), slice(wH, -wH // 2), slice(-wH // 2, None))
            cnt = 0
            for z in z_slices:
                for h in h_slices:
                    img_mask[:, z, h, :, :] = cnt
                    cnt += 1
            mZ, mH, mW = img_mask.shape[1], img_mask.shape[2], img_mask.shape[3]
            img_mask = img_mask.reshape(1, mZ // wZ, wZ, mH // wH, wH, mW // wW, wW, 1)
            img_mask = torch.permute(img_mask, (0, 5, 1, 3, 2, 4, 6, 7))
            mask_windows = img_mask.reshape(-1, self.type_of_windows, wZ * wH * wW)
            attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
            return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

        def forward(self, x, Z, H, W, roll):
            wZ, wH, wW = window_size
            shortcut = x
            x = x.view(x.shape[0], Z, H, W, x.shape[2])
            x = F.pad(x, (0, 0, 0, 0, self.padding_front, self.padding_back), "constant")
            ori_shape = x.shape

            mask = None
            if roll:
                x = torch.roll(x, shifts=[-wZ // 2, -wH // 2, -wW // 2], dims=(1, 2, 3))
                mask = self.gen_mask(x)

            x_window = x.view(x.shape[0], x.shape[1] // wZ, wZ, x.shape[2] // wH, wH,
                              x.shape[3] // wW, wW, x.shape[-1])
            x_window = torch.permute(x_window, (0, 5, 1, 3, 2, 4, 6, 7))
            x_window = x_window.reshape(x_window.shape[1], x_window.shape[2] * x_window.shape[3],
                                        x_window.shape[4], x_window.shape[5], x_window.shape[6], x_window.shape[7])
            x_window = x_window.contiguous().view(x_window.shape[0], x_window.shape[1], wZ * wH * wW,
                                                  x_window.shape[-1])
            attn_windows = self.attention(x_window, mask)

            x_shifted = attn_windows.view(1, attn_windows.shape[0], Z // wZ, H // wH + 1, wZ, wH, wW, -1)
            x_shifted = torch.permute(x_shifted, (0, 2, 4, 3, 5, 1, 6, 7))
            x_shifted = x_shifted.contiguous().view(ori_shape)

            if roll:
                x = torch.roll(x_shifted, shifts=[wZ // 2, wH // 2, wW // 2], dims=(1, 2, 3))
            else:
                x = x_shifted

            x = x[:, :, self.padding_front:x.shape[2] - self.padding_back, :, :]
            x = x.contiguous().view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])

            x = shortcut + self.norm1(x)
            x = x + self.norm2(self.linear(x))
            return x

    class EarthSpecificLayer(nn.Module):
        def __init__(self, depth, dim, heads):
            super().__init__()
            block_list = OrderedDict()
            for i in range(depth):
                block_list[f"EarthSpecificBlock{i}"] = EarthSpecificBlock(dim, heads)
            self.blocks = nn.Sequential(block_list)

        def forward(self, x, Z, H, W):
            for i, blk in enumerate(self.blocks):
                x = blk(x, Z, H, W, roll=(i % 2 == 1))
            return x

    class PatchEmbedding_pretrain(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv = nn.Conv1d(192, dim, 1)
            self.conv_surface = nn.Conv1d(112, dim, 1)

        def forward(self, input, input_surface, statistics, maps, const_h):
            surface_mean, surface_std, upper_mean, upper_std = statistics
            input_surface = input_surface.reshape(input_surface.shape[0], input_surface.shape[1], 1,
                                                  input_surface.shape[-2], input_surface.shape[-1])
            input_surface = torch.permute(input_surface, (0, 2, 3, 4, 1))
            input_surface = (input_surface - surface_mean) / surface_std
            input_surface = torch.permute(input_surface, (0, 4, 1, 2, 3))
            input_surface = input_surface.view(input_surface.shape[0], input_surface.shape[1],
                                               input_surface.shape[-2], input_surface.shape[-1])
            input_surface = F.pad(input_surface, (0, 0, 0, 3), "constant")
            input_surface = torch.cat((input_surface, maps), dim=1)
            input_surface = input_surface.view(input_surface.shape[0], input_surface.shape[1],
                                               input_surface.shape[-2] // 4, 4, input_surface.shape[-1] // 4, 4)
            input_surface = torch.permute(input_surface, (0, 1, 3, 5, 2, 4))
            input_surface = input_surface.reshape(input_surface.shape[0], input_surface.shape[1] *
                                                  input_surface.shape[2] * input_surface.shape[3], -1)
            input_surface = self.conv_surface(input_surface)
            input_surface = input_surface.view(input_surface.shape[0], input_surface.shape[1], 1, 181, 360)

            input = input.reshape(input.shape[0], input.shape[1], 1, input.shape[2], input.shape[-2],
                                  input.shape[-1])
            input = torch.permute(input, (0, 2, 3, 4, 5, 1))
            input = torch.flip(input, [2])
            input = (input - upper_mean) / upper_std
            input = torch.permute(input, (0, 5, 1, 2, 3, 4))
            input = torch.flip(input, [3])
            input = torch.cat((input, const_h), dim=1)
            input = input.reshape(input.shape[0], input.shape[1], input.shape[3], input.shape[-2],
                                  input.shape[-1])
            input = F.pad(input, (0, 0, 0, 3, 0, 1), "constant")
            input = input.reshape(input.shape[0], input.shape[1], input.shape[2] // 2, 2,
                                  input.shape[-2] // 4, 4, input.shape[-1] // 4, 4)
            input = input.permute(0, 1, 3, 5, 7, 2, 4, 6)
            input = input.reshape(input.shape[0], input.shape[1] * input.shape[2] * input.shape[3] *
                                  input.shape[4], -1)
            input = self.conv(input)
            input = input.view(input.shape[0], input.shape[1], 7, 181, 360)

            x = torch.cat((input_surface, input), dim=2)
            x = x.view(x.shape[0], x.shape[1], -1)
            return torch.permute(x, (0, 2, 1))

    class DownSample(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = nn.LayerNorm(4 * dim)

        def forward(self, x, Z, H, W):
            x = x.view(x.shape[0], Z, H, W, x.shape[-1])
            x = F.pad(x, (0, 0, 0, 0, 0, 1), "constant")
            Z, H, W = x.shape[1], x.shape[2], x.shape[3]
            x = x.view(x.shape[0], Z, H // 2, 2, W // 2, 2, x.shape[-1])
            x = torch.permute(x, (0, 1, 2, 4, 3, 5, 6))
            x = x.reshape(x.shape[0], Z * (H // 2) * (W // 2), 4 * x.shape[-1])
            return self.linear(self.norm(x))

    class UpSample(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, output_dim * 4, bias=False)
            self.linear2 = nn.Linear(output_dim, output_dim, bias=False)
            self.norm = nn.LayerNorm(output_dim)

        def forward(self, x):
            x = self.linear1(x)
            x = x.view(x.shape[0], 8, 91, 180, 2, 2, x.shape[-1] // 4)
            x = torch.permute(x, (0, 1, 2, 4, 3, 5, 6))
            x = x.contiguous().view(x.shape[0], 8, 182, 360, x.shape[-1])
            x = x[:, :, :x.shape[2] - 1, :, :]
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[-1])
            return self.linear2(self.norm(x))

    class PatchRecovery_pretrain(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.conv = nn.Conv1d(dim, 160, 1)
            self.conv_surface = nn.Conv1d(dim, 64, 1)

        def forward(self, x, Z, H, W):
            x = torch.permute(x, (0, 2, 1))
            x = x.view(x.shape[0], x.shape[1], Z, H, W)

            output = x[:, :, 1:, :, :]
            output = output.view(output.shape[0], output.shape[1], -1)
            output = self.conv(output)
            output = output.reshape(output.shape[0], 5, 2, 4, 4, Z - 1, H, W)
            output = torch.permute(output, (0, 1, 5, 2, 6, 3, 7, 4))
            output = output.reshape(output.shape[0], 5, 14, 724, 1440)
            output = output[:, :, :13, :721, :]

            output_surface = x[:, :, 0, :, :]
            output_surface = output_surface.view(output_surface.shape[0], self.dim, -1)
            output_surface = self.conv_surface(output_surface)
            output_surface = output_surface.view(output_surface.shape[0], 4, 4, 4, H, W)
            output_surface = torch.permute(output_surface, (0, 1, 4, 2, 5, 3))
            output_surface = output_surface.reshape(output_surface.shape[0], 4, 724, 1440)
            output_surface = output_surface[:, :, :721, :]
            return output, output_surface

    class PanguModel(nn.Module):
        def __init__(self, depths=(2, 6, 6, 2), num_heads=(6, 12, 12, 6), dims=(192, 384, 384, 192)):
            super().__init__()
            self._input_layer = PatchEmbedding_pretrain(dims[0])
            self.downsample = DownSample(dims[0])
            layer_list = OrderedDict()
            for i in range(4):
                layer_list[f"EarthSpecificLayer{i}"] = EarthSpecificLayer(depths[i], dims[i], num_heads[i])
            self.layers = nn.Sequential(layer_list)
            self.upsample = UpSample(dims[-2], dims[-1])
            self._output_layer = PatchRecovery_pretrain(dims[-2])

        def forward(self, input, input_surface, statistics, maps, const_h):
            x = self._input_layer(input, input_surface, statistics, maps, const_h)
            x = self.layers[0](x, 8, 181, 360)
            skip = x
            x = self.downsample(x, 8, 181, 360)
            x = self.layers[1](x, 8, 91, 180)
            x = self.layers[2](x, 8, 91, 180)
            x = self.upsample(x)
            x = self.layers[3](x, 8, 181, 360)
            x = torch.cat((skip, x), dim=-1)
            return self._output_layer(x, 8, 181, 360)

    import gc

    zip_path = os.path.join(DEFAULT_CACHE_DIR, "pangu_pretrained_model",
                             "pretrained_model", "pangu_weather_24_torch.pth")

    rng = np.random.RandomState(0)
    raw_upper = (rng.randn(1, 5, 13, 721, 1440) * 0.1).astype("float32")
    raw_surface = (rng.randn(1, 4, 721, 1440) * 0.1).astype("float32")
    const_h = (rng.randn(1, 1, 1, 13, 721, 1440) * 0.1).astype("float32")
    maps_721 = (rng.randn(1, 3, 721, 1440) * 0.1).astype("float32")
    maps = np.pad(maps_721, ((0, 0), (0, 0), (0, 3), (0, 0)), mode="constant")

    torch_model = PanguModel()
    ckpt = torch.load(zip_path, map_location="cpu", weights_only=False)
    missing, unexpected = torch_model.load_state_dict(ckpt["model"], strict=False)
    assert not missing and not unexpected
    torch_model.eval()

    zero_s, one_s = torch.zeros(1, 1, 1, 1, 4), torch.ones(1, 1, 1, 1, 4)
    zero_u, one_u = torch.zeros(1, 1, 1, 1, 1, 5), torch.ones(1, 1, 1, 1, 1, 5)

    with torch.no_grad():
        torch_out, torch_out_surface = torch_model(
            torch.from_numpy(raw_upper), torch.from_numpy(raw_surface),
            [zero_s, one_s, zero_u, one_u], torch.from_numpy(maps), torch.from_numpy(const_h))
    torch_out, torch_out_surface = torch_out.numpy(), torch_out_surface.numpy()

    del torch_model, ckpt, missing, unexpected
    gc.collect()

    upper_6ch = np.concatenate(
        [np.transpose(raw_upper[0], (1, 2, 3, 0)), const_h[0, 0, 0][:, :, :, None]], axis=-1)[None]
    surface_7ch = np.concatenate(
        [np.transpose(raw_surface[0], (1, 2, 0)), np.transpose(maps_721[0], (1, 2, 0))], axis=-1)[None]
    del raw_upper, raw_surface, const_h, maps, maps_721
    gc.collect()

    keras_model, report = pangu_weather_24()
    assert not report["unused_source_keys"]
    assert all(k.endswith("attn_mask") for k in report["missing_in_source"])

    with torch.no_grad():
        keras_out, keras_out_surface = keras_model([upper_6ch, surface_7ch], training=False)
    keras_out, keras_out_surface = keras_out.numpy(), keras_out_surface.numpy()

    max_diff_upper = np.abs(torch_out - keras_out).max()
    max_diff_surface = np.abs(torch_out_surface - keras_out_surface).max()
    assert max_diff_upper < 1e-2, f"PanguWeather upper output mismatch: max abs diff {max_diff_upper}"
    assert max_diff_surface < 1e-2, f"PanguWeather surface output mismatch: max abs diff {max_diff_surface}"
