"""Build/shape sanity checks + PyTorch weight-port round-trip test for
`keras_climate.remote_sensing.ringmo` (RingMo / RingMoEncoder).

Run with: pytest keras_climate/remote_sensing/test_ringmo.py
"""
import numpy as np
import pytest
import keras

from keras_climate.remote_sensing.ringmo import (
    RingMo, RingMoEncoder, SimMIMDecoder, PIMask, window_partition, window_reverse,
)
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_ringmo_mapper


# --------------------------------------------------------------------------
# Build / forward-pass sanity checks (Keras only, no torch required)
# --------------------------------------------------------------------------

def test_builds_and_runs():
    model = RingMo(img_size=48, in_chans=3, embed_dim=16, depths=(2, 2, 2),
                    num_heads=(2, 4, 8), window_size=3, mlp_ratio=2.0, mask_patch_size=16)
    x = np.random.randn(2, 48, 48, 3).astype("float32")
    pred, mask = model(x)
    assert tuple(pred.shape) == (2, 48, 48, 3)
    assert tuple(mask.shape) == (2, 48, 48, 1)


def test_encoder_alone_for_downstream_tasks():
    encoder = RingMoEncoder(img_size=48, embed_dim=16, depths=(2, 2, 2),
                             num_heads=(2, 4, 8), window_size=3, mlp_ratio=2.0)
    x = np.random.randn(2, 48, 48, 3).astype("float32")
    tokens = encoder(x)
    final_grid = 48 // 4 // 4  # /4 patch embed, /2 per downsample x2 downsamples
    assert tuple(tokens.shape) == (2, final_grid * final_grid, 16 * 4)


def test_pi_mask_only_zeros_a_fraction_of_pixels_per_block():
    """Regression check for the whole point of PI-Mask: within a masked
    block, NOT every pixel should be zeroed (unlike vanilla SimMIM block
    masking) - with `inside_ratio < 1`, some pixels inside masked blocks
    survive."""
    x = np.ones((4, 32, 32, 3), dtype="float32")
    masked, mask = PIMask(mask_patch_size=32, mask_ratio=1.0, inside_ratio=0.5)(x)
    masked = keras.ops.convert_to_numpy(masked)
    # mask_ratio=1.0 -> every block is "selected"; inside_ratio=0.5 means
    # roughly half the pixels within it actually get zeroed - so the
    # fraction of surviving (nonzero) pixels should be well away from 0.
    frac_kept = (masked != 0).mean()
    assert 0.2 < frac_kept < 0.8


def test_window_partition_reverse_roundtrip():
    x = np.random.randn(2, 12, 12, 4).astype("float32")
    windows = window_partition(x, window_size=3)
    assert tuple(windows.shape) == (2 * 16, 3, 3, 4)
    back = keras.ops.convert_to_numpy(window_reverse(windows, 3, 12, 12))
    assert np.allclose(back, x)


# --------------------------------------------------------------------------
# PyTorch weight-port round-trip against a from-scratch reference matching
# this repo's own RingMo naming/architecture (no official checkpoint
# exists - see module docstring).
# --------------------------------------------------------------------------

def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    def window_partition_np(x, window_size):
        B, H, W, C = x.shape
        x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, window_size, window_size, C)

    def window_reverse_np(windows, window_size, H, W):
        C = windows.shape[-1]
        nh, nw = H // window_size, W // window_size
        B = windows.shape[0] // (nh * nw)
        x = windows.reshape(B, nh, nw, window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, H, W, C)

    class TorchWindowAttention(nn.Module):
        def __init__(self, dim, window_size, num_heads):
            super().__init__()
            self.window_size = window_size
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)

            coords = np.stack(np.meshgrid(np.arange(window_size), np.arange(window_size), indexing="ij"))
            coords_flat = coords.reshape(2, -1)
            rel = coords_flat[:, :, None] - coords_flat[:, None, :]
            rel = rel.transpose(1, 2, 0)
            rel[:, :, 0] += window_size - 1
            rel[:, :, 1] += window_size - 1
            rel[:, :, 0] *= 2 * window_size - 1
            self.register_buffer("rel_pos_index", torch.from_numpy(rel.sum(-1).astype("int64")).reshape(-1))
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) ** 2, num_heads))

        def forward(self, x, mask=None):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            attn = q @ k.transpose(-2, -1)

            bias = self.relative_position_bias_table[self.rel_pos_index].reshape(N, N, self.num_heads)
            bias = bias.permute(2, 0, 1)
            attn = attn + bias.unsqueeze(0)

            if mask is not None:
                nw = mask.shape[0]
                attn = attn.reshape(B_ // nw, nw, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.reshape(B_, self.num_heads, N, N)

            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            return self.proj(out)

    class TorchMlp(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))

    class TorchSwinBlock(nn.Module):
        def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, mlp_ratio):
            super().__init__()
            H, W = input_resolution
            if min(H, W) <= window_size:
                shift_size = 0
                window_size = min(H, W)
            self.input_resolution = (H, W)
            self.dim = dim
            self.window_size = window_size
            self.shift_size = shift_size
            self.norm1 = nn.LayerNorm(dim, eps=1e-5)
            self.attn = TorchWindowAttention(dim, window_size, num_heads)
            self.norm2 = nn.LayerNorm(dim, eps=1e-5)
            self.mlp = TorchMlp(dim, int(dim * mlp_ratio))

            if shift_size > 0:
                img_mask = np.zeros((1, H, W, 1), dtype="float32")
                slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
                cnt = 0
                for h in slices:
                    for w in slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                m = torch.from_numpy(img_mask)
                mw = window_partition_np(m, window_size).reshape(-1, window_size * window_size)
                attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
                self.register_buffer("attn_mask", attn_mask)
            else:
                self.attn_mask = None

        def forward(self, x):
            H, W = self.input_resolution
            B, C = x.shape[0], self.dim
            shortcut = x
            x = self.norm1(x).reshape(B, H, W, C)
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            windows = window_partition_np(x, self.window_size).reshape(-1, self.window_size ** 2, C)
            attn_out = self.attn(windows, mask=self.attn_mask)
            attn_out = attn_out.reshape(-1, self.window_size, self.window_size, C)
            x = window_reverse_np(attn_out, self.window_size, H, W)
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x = x.reshape(B, H * W, C)
            x = shortcut + x
            x = x + self.mlp(self.norm2(x))
            return x

    class TorchPatchMerging(nn.Module):
        def __init__(self, input_resolution, dim):
            super().__init__()
            self.input_resolution = input_resolution
            self.dim = dim
            self.norm = nn.LayerNorm(4 * dim, eps=1e-5)
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        def forward(self, x):
            H, W = self.input_resolution
            B, C = x.shape[0], self.dim
            x = x.reshape(B, H, W, C)
            x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], dim=-1).reshape(B, (H // 2) * (W // 2), 4 * C)
            return self.reduction(self.norm(x))

    class TorchStage(nn.Module):
        def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio, downsample):
            super().__init__()
            self.blocks = nn.ModuleList([
                TorchSwinBlock(dim, input_resolution, num_heads, window_size,
                               0 if i % 2 == 0 else window_size // 2, mlp_ratio)
                for i in range(depth)
            ])
            self.downsample = TorchPatchMerging(input_resolution, dim) if downsample else None

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            if self.downsample is not None:
                x = self.downsample(x)
            return x

    class TorchConvBN(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_c, eps=1e-5)

        def forward(self, x):
            return F.gelu(self.bn(self.conv(x)))

    class TorchPatchEmbed(nn.Module):
        def __init__(self, in_chans, embed_dim):
            super().__init__()
            mid = embed_dim // 2
            self.conv1 = TorchConvBN(in_chans, mid)
            self.conv2 = TorchConvBN(mid, mid)
            self.conv3 = nn.Conv2d(mid, embed_dim, 1)
            self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            return self.norm(x), H, W

    class TorchRingMoEncoder(nn.Module):
        def __init__(self, img_size, in_chans, embed_dim, depths, num_heads, window_size, mlp_ratio):
            super().__init__()
            self.patch_embed = TorchPatchEmbed(in_chans, embed_dim)
            grid = img_size // 4
            dim, res = embed_dim, grid
            stages = []
            for i, (depth, heads) in enumerate(zip(depths, num_heads)):
                downsample = i < len(depths) - 1
                stages.append(TorchStage(dim, (res, res), depth, heads, window_size, mlp_ratio, downsample))
                if downsample:
                    dim *= 2
                    res //= 2
            self.stages = nn.ModuleList(stages)
            self.norm = nn.LayerNorm(dim, eps=1e-5)
            self.final_dim = dim
            self.final_grid = res

        def forward(self, x):
            tokens, H, W = self.patch_embed(x)
            for stage in self.stages:
                tokens = stage(tokens)
            return self.norm(tokens)

    class TorchSimMIMDecoder(nn.Module):
        def __init__(self, dim, encoder_stride, in_chans):
            super().__init__()
            self.decoder = nn.Conv2d(dim, in_chans * encoder_stride ** 2, 1)
            self.encoder_stride = encoder_stride

        def forward(self, x):
            x = self.decoder(x)
            return F.pixel_shuffle(x, self.encoder_stride)

    class TorchRingMoEncoderDecoder(nn.Module):
        def __init__(self, img_size, in_chans, embed_dim, depths, num_heads, window_size, mlp_ratio):
            super().__init__()
            self.encoder = TorchRingMoEncoder(img_size, in_chans, embed_dim, depths, num_heads,
                                               window_size, mlp_ratio)
            encoder_stride = 4 * (2 ** (len(depths) - 1))
            self.decoder = TorchSimMIMDecoder(self.encoder.final_dim, encoder_stride, in_chans)

        def forward(self, x):
            tokens = self.encoder(x)
            B = tokens.shape[0]
            g = self.encoder.final_grid
            feat = tokens.transpose(1, 2).reshape(B, self.encoder.final_dim, g, g)
            return self.decoder(feat)

    torch.manual_seed(0)
    img_size, in_chans, embed_dim = 48, 3, 16
    depths, num_heads, window_size, mlp_ratio = (2, 2, 2), (2, 4, 8), 3, 2.0

    torch_model = TorchRingMoEncoderDecoder(img_size, in_chans, embed_dim, depths, num_heads,
                                             window_size, mlp_ratio)
    torch_model.eval()
    with torch.no_grad():
        for m in torch_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_(0.0, 0.1)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
        for name, p in torch_model.named_parameters():
            if "weight" in name and p.ndim >= 2 and "bn" not in name:
                p.normal_(0.0, 0.2)
            elif p.ndim <= 1 and "bn" not in name:
                p.normal_(0.0, 0.05)

    # Strip the reference wrapper's "encoder."/"decoder." prefixes (its own
    # `TorchSimMIMDecoder.decoder` conv submodule is itself named "decoder",
    # so this also correctly collapses "decoder.decoder.*" -> "decoder.*").
    state_dict = {}
    for k, v in torch_model.state_dict().items():
        if "num_batches_tracked" in k or "rel_pos_index" in k or "attn_mask" in k:
            continue
        if k.startswith("encoder."):
            state_dict[k[len("encoder."):]] = v.detach().numpy()
        elif k.startswith("decoder."):
            state_dict[k[len("decoder."):]] = v.detach().numpy()

    encoder = RingMoEncoder(img_size=img_size, embed_dim=embed_dim, depths=depths,
                             num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                             name="ringmo_encoder")
    x0 = np.zeros((1, img_size, img_size, in_chans), dtype="float32")
    tok0 = encoder(x0)
    decoder = SimMIMDecoder(4 * (2 ** (len(depths) - 1)), in_chans, name="ringmo_decoder")
    feat0 = keras.layers.Reshape((encoder.final_grid, encoder.final_grid, encoder.final_dim))(tok0)
    decoder(feat0)  # build

    mapper = build_ringmo_mapper(depths)
    enc_report = WeightConverter(encoder, state_dict, mapper).convert(strict=False, verbose=False)
    dec_report = WeightConverter(decoder, state_dict, mapper).convert(strict=False, verbose=False)
    # `attn/rel_pos_index` and `attn_mask` are non-trainable buffers derived
    # purely from `window_size`/`shift_size` config (see `WindowAttention`/
    # `SwinTransformerBlock`), not real checkpoint weights - they have no
    # source-key counterpart by design.
    assert all(k.endswith(("rel_pos_index", "attn_mask")) for k in enc_report["missing_in_source"])
    assert not dec_report["missing_in_source"]

    x_np = np.random.randn(1, in_chans, img_size, img_size).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    tokens = encoder(keras_in, training=False)
    feat = keras.layers.Reshape((encoder.final_grid, encoder.final_grid, encoder.final_dim))(tokens)
    keras_out = keras.ops.convert_to_numpy(decoder(feat))
    keras_out = np.transpose(keras_out, (0, 3, 1, 2))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"RingMo weight port numerical mismatch: max abs diff {max_diff}"
