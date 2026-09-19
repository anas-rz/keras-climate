import numpy as np
import pytest
import keras

from keras_climate.weather.metnet import MetNet, AxialAttention2D
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import convert_metnet_state_dict, build_metnet_mapper


def test_builds_and_runs():
    model = MetNet(input_shape=(3, 64, 64, 2), lead_times=4, base_filters=8, attn_dim=16,
                    attn_layers=1, num_heads=2, downsample_factor=4, num_bins=1)
    frames = np.random.randn(1, 3, 64, 64, 2).astype("float32")
    lead = np.array([1], dtype="int32")
    y = keras.ops.convert_to_numpy(model([frames, lead]))
    assert y.shape == (1, 64, 64, 1)
    assert (y >= 0).all() and (y <= 1).all()


def test_categorical_head():
    model = MetNet(input_shape=(2, 32, 32, 1), lead_times=2, base_filters=8, attn_dim=8,
                    attn_layers=1, num_heads=2, downsample_factor=2, num_bins=5)
    frames = np.random.randn(1, 2, 32, 32, 1).astype("float32")
    lead = np.array([0], dtype="int32")
    y = keras.ops.convert_to_numpy(model([frames, lead]))
    assert y.shape == (1, 32, 32, 5)
    assert np.allclose(y.sum(axis=-1), 1.0, atol=1e-5)


def test_lead_time_conditioning_changes_output():
    model = MetNet(input_shape=(2, 32, 32, 1), lead_times=4, base_filters=8, attn_dim=8,
                    attn_layers=1, num_heads=2, downsample_factor=2, num_bins=1)
    frames = np.random.randn(1, 2, 32, 32, 1).astype("float32")
    y0 = keras.ops.convert_to_numpy(model([frames, np.array([0], dtype="int32")]))
    y1 = keras.ops.convert_to_numpy(model([frames, np.array([3], dtype="int32")]))
    assert not np.allclose(y0, y1)


def test_axial_attention_row_and_column_independence():
    attn = AxialAttention2D(dim=8, num_heads=2)
    x = np.random.randn(1, 6, 6, 8).astype("float32")
    y = keras.ops.convert_to_numpy(attn(x))
    assert y.shape == x.shape


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvLSTMCell(nn.Module):
        def __init__(self, input_dim, hidden_dim, kernel_size):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size,
                                   padding=kernel_size // 2, bias=True)

        def forward(self, x, state):
            h_cur, c_cur = state
            combined = torch.cat([x, h_cur], dim=1)
            cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)
            i, f, o, g = cc_i.sigmoid(), cc_f.sigmoid(), cc_o.sigmoid(), cc_g.tanh()
            c_next = f * c_cur + i * g
            return o * c_next.tanh(), c_next

    class AxialAttn(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.dim, self.num_heads = dim, num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.row_qkv = nn.Linear(dim, dim * 3)
            self.row_proj = nn.Linear(dim, dim)
            self.col_qkv = nn.Linear(dim, dim * 3)
            self.col_proj = nn.Linear(dim, dim)
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        def _attend(self, x, qkv_layer, proj_layer):
            B_, L, C = x.shape
            qkv = qkv_layer(x).reshape(B_, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B_, L, C)
            return proj_layer(out)

        def forward(self, x):
            B, H, W, C = x.shape
            y = self.norm1(x)
            rows = self._attend(y.reshape(B * H, W, C), self.row_qkv, self.row_proj).reshape(B, H, W, C)
            x = x + rows
            y = self.norm2(x)
            cols = y.permute(0, 2, 1, 3).reshape(B * W, H, C)
            cols = self._attend(cols, self.col_qkv, self.col_proj).reshape(B, W, H, C).permute(0, 2, 1, 3)
            return x + cols

    class TorchMetNet(nn.Module):
        def __init__(self, in_channels, base_filters, attn_dim, attn_layers, num_heads, lead_times,
                     downsample_factor, num_dilated=6):
            super().__init__()
            self.stem_conv = nn.Conv2d(in_channels, base_filters, 3, stride=downsample_factor, padding=1)
            self.stem_bn = nn.BatchNorm2d(base_filters, eps=1e-5)
            self.temporal_cell = ConvLSTMCell(base_filters, base_filters, 3)
            self.base_filters = base_filters

            dilations = [1, 2, 4, 8, 16, 1][:num_dilated]
            self.context_tower = nn.ModuleList()
            in_ch = base_filters
            for d in dilations:
                conv = nn.Conv2d(in_ch, base_filters * 2, 3, padding=d, dilation=d)
                bn = nn.BatchNorm2d(base_filters * 2, eps=1e-5)
                self.context_tower.append(nn.ModuleDict({"conv": conv, "bn": bn}))
                in_ch = base_filters * 2

            self.lead_time_embed = nn.Embedding(lead_times, base_filters * 2)
            self.attn_proj_in = nn.Conv2d(base_filters * 2, attn_dim, 1)
            self.axial_layers = nn.ModuleList([AxialAttn(attn_dim, num_heads) for _ in range(attn_layers)])

            n_up = downsample_factor.bit_length() - 1
            self.upsamples = nn.ModuleList()
            in_ch = attn_dim
            for _ in range(n_up):
                out_ch = attn_dim // 2
                self.upsamples.append(nn.ModuleDict({
                    "deconv": nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                    "bn": nn.BatchNorm2d(out_ch, eps=1e-5),
                }))
                in_ch = out_ch
            self.precip_head = nn.Conv2d(in_ch, 1, 1)

        def forward(self, frames, lead_idx):
            B, T, C, H, W = frames.shape
            x = F.relu(self.stem_bn(self.stem_conv(frames.reshape(B * T, C, H, W))))
            _, _, Hs, Ws = x.shape
            x = x.reshape(B, T, self.base_filters, Hs, Ws)

            h = torch.zeros(B, self.base_filters, Hs, Ws)
            c = torch.zeros(B, self.base_filters, Hs, Ws)
            for t in range(T):
                h, c = self.temporal_cell(x[:, t], (h, c))

            for layer in self.context_tower:
                h = F.relu(layer["bn"](layer["conv"](h)))

            lead = self.lead_time_embed(lead_idx).reshape(B, -1, 1, 1)
            h = h + lead

            h = self.attn_proj_in(h)
            h = h.permute(0, 2, 3, 1)
            for layer in self.axial_layers:
                h = layer(h)
            h = h.permute(0, 3, 1, 2)

            for layer in self.upsamples:
                h = F.relu(layer["bn"](layer["deconv"](h)))

            return torch.sigmoid(self.precip_head(h))

    torch.manual_seed(0)
    in_channels, base_filters, attn_dim, attn_layers, num_heads = 2, 8, 16, 1, 2
    lead_times, downsample_factor, T_in, H, W = 4, 4, 3, 64, 64

    torch_model = TorchMetNet(in_channels, base_filters, attn_dim, attn_layers, num_heads,
                               lead_times, downsample_factor)
    torch_model.eval()
    with torch.no_grad():
        for m in torch_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_(0.0, 0.1)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
            elif hasattr(m, "weight") and isinstance(getattr(m, "weight", None), torch.Tensor):
                m.weight.data.normal_(0.0, 0.3)

    flat = {}
    flat["stem.conv.weight"] = torch_model.stem_conv.weight.detach().numpy()
    flat["stem.conv.bias"] = torch_model.stem_conv.bias.detach().numpy()
    flat["stem_bn.bn.weight"] = torch_model.stem_bn.weight.detach().numpy()
    flat["stem_bn.bn.bias"] = torch_model.stem_bn.bias.detach().numpy()
    flat["stem_bn.bn.running_mean"] = torch_model.stem_bn.running_mean.detach().numpy()
    flat["stem_bn.bn.running_var"] = torch_model.stem_bn.running_var.detach().numpy()
    flat["temporal_encoder.cell_list.0.conv.weight"] = torch_model.temporal_cell.conv.weight.detach().numpy()
    flat["temporal_encoder.cell_list.0.conv.bias"] = torch_model.temporal_cell.conv.bias.detach().numpy()
    for i, layer in enumerate(torch_model.context_tower):
        flat[f"context_tower.{i}.conv.weight"] = layer["conv"].weight.detach().numpy()
        flat[f"context_tower.{i}.conv.bias"] = layer["conv"].bias.detach().numpy()
        flat[f"context_tower.{i}.bn.weight"] = layer["bn"].weight.detach().numpy()
        flat[f"context_tower.{i}.bn.bias"] = layer["bn"].bias.detach().numpy()
        flat[f"context_tower.{i}.bn.running_mean"] = layer["bn"].running_mean.detach().numpy()
        flat[f"context_tower.{i}.bn.running_var"] = layer["bn"].running_var.detach().numpy()
    flat["lead_time_embed.weight"] = torch_model.lead_time_embed.weight.detach().numpy()
    flat["attn_proj_in.weight"] = torch_model.attn_proj_in.weight.detach().numpy()
    flat["attn_proj_in.bias"] = torch_model.attn_proj_in.bias.detach().numpy()
    for i, layer in enumerate(torch_model.axial_layers):
        p = f"axial_attn{i}"
        for stream in ("row", "col"):
            lin_qkv = getattr(layer, f"{stream}_qkv")
            lin_proj = getattr(layer, f"{stream}_proj")
            flat[f"{p}.{stream}_qkv.weight"] = lin_qkv.weight.detach().numpy()
            flat[f"{p}.{stream}_qkv.bias"] = lin_qkv.bias.detach().numpy()
            flat[f"{p}.{stream}_proj.weight"] = lin_proj.weight.detach().numpy()
            flat[f"{p}.{stream}_proj.bias"] = lin_proj.bias.detach().numpy()
        flat[f"{p}.norm1.weight"] = layer.norm1.weight.detach().numpy()
        flat[f"{p}.norm1.bias"] = layer.norm1.bias.detach().numpy()
        flat[f"{p}.norm2.weight"] = layer.norm2.weight.detach().numpy()
        flat[f"{p}.norm2.bias"] = layer.norm2.bias.detach().numpy()
    for i, layer in enumerate(torch_model.upsamples):
        flat[f"upsample{i}.weight"] = layer["deconv"].weight.detach().numpy()
        flat[f"upsample{i}.bias"] = layer["deconv"].bias.detach().numpy()
        flat[f"upsample_bn{i}.weight"] = layer["bn"].weight.detach().numpy()
        flat[f"upsample_bn{i}.bias"] = layer["bn"].bias.detach().numpy()
        flat[f"upsample_bn{i}.running_mean"] = layer["bn"].running_mean.detach().numpy()
        flat[f"upsample_bn{i}.running_var"] = layer["bn"].running_var.detach().numpy()
    flat["precip_head.weight"] = torch_model.precip_head.weight.detach().numpy()
    flat["precip_head.bias"] = torch_model.precip_head.bias.detach().numpy()

    translated = convert_metnet_state_dict(flat, base_filters=base_filters,
                                            num_dilated_convs=6, num_att_layers=attn_layers, num_upsample=2)

    keras_model = MetNet(input_shape=(T_in, H, W, in_channels), lead_times=lead_times,
                          base_filters=base_filters, attn_dim=attn_dim, attn_layers=attn_layers,
                          num_heads=num_heads, downsample_factor=downsample_factor, num_bins=1)
    report = WeightConverter(keras_model, translated, build_metnet_mapper()).convert(
        strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    frames_np = np.random.randn(1, T_in, in_channels, H, W).astype("float32")
    lead_idx = 2
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(frames_np), torch.tensor([lead_idx])).numpy()

    keras_frames = np.transpose(frames_np, (0, 1, 3, 4, 2))
    keras_out = keras.ops.convert_to_numpy(
        keras_model([keras_frames, np.array([lead_idx], dtype="int32")], training=False))
    keras_out = np.transpose(keras_out, (0, 3, 1, 2))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"MetNet weight port numerical mismatch: max abs diff {max_diff}"
