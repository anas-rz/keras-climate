"""Build/shape sanity checks + PyTorch weight-port round-trip test for
`keras_climate.remote_sensing.segformer.SegFormer`.

Run with: pytest keras_climate/remote_sensing/test_segformer.py
"""
import numpy as np
import pytest
import keras

from .segformer import SegFormer, MIT_CONFIGS
from ..weights import WeightConverter
from ..weights.mappings import build_segformer_mapper, build_segformer_param_kind_map
from ..weights.pretrained import segformer_b0_ade20k


# --------------------------------------------------------------------------
# Build / forward-pass sanity checks (Keras only, no torch required)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(MIT_CONFIGS))
def test_builds_and_runs(variant):
    model = SegFormer(input_shape=(128, 128, 3), num_classes=5, variant=variant)
    x = np.random.randn(1, 128, 128, 3).astype("float32")
    y = model(x)
    assert tuple(y.shape) == (1, 128, 128, 5)


def test_non_multiple_of_32_input_size():
    """Regression check for the MiT encoder's overlap-patch-embed strides
    (4, 2, 2, 2 -> total downsample 32) and decode-head upsampling."""
    model = SegFormer(input_shape=(150, 150, 3), num_classes=2, variant="b0")
    x = np.random.randn(1, 150, 150, 3).astype("float32")
    y = model(x)
    assert tuple(y.shape) == (1, 150, 150, 2)


def test_final_activation_applied():
    model = SegFormer(input_shape=(64, 64, 3), num_classes=1, variant="b0",
                       final_activation="sigmoid")
    x = np.random.randn(1, 64, 64, 3).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.min() >= 0.0 and y.max() <= 1.0


# --------------------------------------------------------------------------
# PyTorch weight-port round-trip: a reference MiT backbone + all-MLP decode
# head matching the official NVlabs/SegFormer state_dict naming (the
# convention `build_segformer_mapper` assumes) is built, its (random)
# weights are ported through the real converter + mapping, and the two
# forward passes are compared numerically end-to-end.
# --------------------------------------------------------------------------

def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class DWConv(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

        def forward(self, x, H, W):
            B, N, C = x.shape
            x = x.transpose(1, 2).view(B, C, H, W)
            x = self.dwconv(x)
            return x.flatten(2).transpose(1, 2)

    class MixFFN(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.dwconv = DWConv(hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, dim)

        def forward(self, x, H, W):
            x = self.fc1(x)
            x = self.dwconv(x, H, W)
            x = self.act(x)
            return self.fc2(x)

    class EfficientAttention(nn.Module):
        def __init__(self, dim, num_heads, sr_ratio):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.sr_ratio = sr_ratio
            self.q = nn.Linear(dim, dim, bias=True)
            self.kv = nn.Linear(dim, dim * 2, bias=True)
            self.proj = nn.Linear(dim, dim)
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)

        def forward(self, x, H, W):
            B, N, C = x.shape
            q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
            else:
                x_ = x
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(out)

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio, sr_ratio):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = EfficientAttention(dim, num_heads, sr_ratio)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = MixFFN(dim, int(dim * mlp_ratio))

        def forward(self, x, H, W):
            x = x + self.attn(self.norm1(x), H, W)
            x = x + self.mlp(self.norm2(x), H, W)
            return x

    class OverlapPatchEmbed(nn.Module):
        def __init__(self, patch_size, stride, in_ch, embed_dim):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride,
                                   padding=patch_size // 2)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
            x = self.proj(x)
            H, W = x.shape[2], x.shape[3]
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            return x, H, W

    class MixVisionTransformer(nn.Module):
        def __init__(self, cfg, in_chans=3):
            super().__init__()
            patch_sizes = [7, 3, 3, 3]
            strides = [4, 2, 2, 2]
            dims = cfg["embed_dims"]
            in_chs = [in_chans] + dims[:-1]
            for s in range(1, 5):
                setattr(self, f"patch_embed{s}",
                        OverlapPatchEmbed(patch_sizes[s - 1], strides[s - 1], in_chs[s - 1], dims[s - 1]))
                blocks = nn.ModuleList([
                    Block(dims[s - 1], cfg["num_heads"][s - 1], 4.0, cfg["sr_ratios"][s - 1])
                    for _ in range(cfg["depths"][s - 1])
                ])
                setattr(self, f"block{s}", blocks)
                setattr(self, f"norm{s}", nn.LayerNorm(dims[s - 1]))

        def forward(self, x):
            B = x.shape[0]
            features = []
            for s in range(1, 5):
                patch_embed = getattr(self, f"patch_embed{s}")
                blocks = getattr(self, f"block{s}")
                norm = getattr(self, f"norm{s}")
                x, H, W = patch_embed(x)
                for blk in blocks:
                    x = blk(x, H, W)
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                features.append(x)
            return features

    class MLPProj(nn.Module):
        def __init__(self, in_dim, embed_dim):
            super().__init__()
            self.proj = nn.Linear(in_dim, embed_dim)

        def forward(self, x):
            x = x.flatten(2).transpose(1, 2)
            return self.proj(x)

    class SegFormerHead(nn.Module):
        def __init__(self, cfg, num_classes):
            super().__init__()
            dims = cfg["embed_dims"]
            decoder_dim = cfg["decoder_dim"]
            self.linear_c4 = MLPProj(dims[3], decoder_dim)
            self.linear_c3 = MLPProj(dims[2], decoder_dim)
            self.linear_c2 = MLPProj(dims[1], decoder_dim)
            self.linear_c1 = MLPProj(dims[0], decoder_dim)
            self.linear_fuse = nn.Module()
            self.linear_fuse.conv = nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False)
            self.linear_fuse.bn = nn.BatchNorm2d(decoder_dim, eps=1e-5)
            self.linear_pred = nn.Conv2d(decoder_dim, num_classes, 1)

        def forward(self, features, target_hw):
            c1, c2, c3, c4 = features
            _, _, h4, w4 = c1.shape  # target = stage-1 resolution (H/4, W/4)

            def proj_upsample(feat, layer):
                B, C, H, W = feat.shape
                t = feat.flatten(2).transpose(1, 2)
                t = layer.proj(t)
                t = t.transpose(1, 2).reshape(B, -1, H, W)
                return F.interpolate(t, size=(h4, w4), mode="bilinear", align_corners=False)

            p4 = proj_upsample(c4, self.linear_c4)
            p3 = proj_upsample(c3, self.linear_c3)
            p2 = proj_upsample(c2, self.linear_c2)
            p1 = proj_upsample(c1, self.linear_c1)

            x = torch.cat([p4, p3, p2, p1], dim=1)
            x = self.linear_fuse.conv(x)
            x = self.linear_fuse.bn(x)
            x = F.relu(x)
            x = self.linear_pred(x)
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            return x

    class TorchSegFormer(nn.Module):
        def __init__(self, cfg, num_classes=5, in_chans=3):
            super().__init__()
            self.backbone = MixVisionTransformer(cfg, in_chans)
            self.decode_head = SegFormerHead(cfg, num_classes)

        def forward(self, x):
            target_hw = x.shape[2], x.shape[3]
            features = self.backbone(x)
            return self.decode_head(features, target_hw)

    torch.manual_seed(0)
    cfg = MIT_CONFIGS["b0"]
    torch_model = TorchSegFormer(cfg, num_classes=5)
    torch_model.eval()
    with torch.no_grad():
        for m in torch_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_(0.0, 0.1)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)

    state_dict = {}
    for k, v in torch_model.state_dict().items():
        if "num_batches_tracked" in k:
            continue
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        state_dict[k] = v.detach().numpy()

    keras_model = SegFormer(input_shape=(256, 256, 3), num_classes=5, variant="b0")
    keras_model(np.zeros((1, 256, 256, 3), dtype="float32"))

    mapper = build_segformer_mapper(cfg)
    param_kind_map = build_segformer_param_kind_map(cfg)
    report = WeightConverter(keras_model, state_dict, mapper,
                              param_kind_map=param_kind_map).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(1, 3, 256, 256).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    keras_out = keras.ops.convert_to_numpy(keras_model(keras_in, training=False))
    keras_out = np.transpose(keras_out, (0, 3, 1, 2))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"SegFormer weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_ade20k_checkpoint():
    """Downloads the real `nvidia/segformer-b0-finetuned-ade-512-512`
    HuggingFace checkpoint (full encoder + decode head, fine-tuned on
    ADE20K) and confirms every weight loads cleanly. Run explicitly with
    `pytest -m pretrained` (network + `transformers` + HF Hub download,
    cached after the first run)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model, report = segformer_b0_ade20k(input_shape=(256, 256, 3))
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x = np.random.rand(1, 256, 256, 3).astype("float32")
    y = keras.ops.convert_to_numpy(model(x, training=False))
    assert y.shape == (1, 256, 256, 150)  # ADE20K has 150 classes
    assert np.isfinite(y).all()
