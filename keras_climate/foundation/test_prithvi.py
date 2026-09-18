"""Build/shape sanity checks + PyTorch weight-port round-trip test for
`keras_climate.foundation.prithvi`.

Run with: pytest keras_climate/foundation/test_prithvi.py
"""
import numpy as np
import pytest
import keras

from keras_climate.foundation.prithvi import PrithviEncoder, PrithviClassifier, PrithviSegmenter, get_3d_sincos_pos_embed
from keras_climate.weights import WeightConverter, load_torch_state_dict_as_numpy
from keras_climate.weights.mappings import build_vit_mapper


# --------------------------------------------------------------------------
# Build / forward-pass sanity checks (Keras only, no torch required)
# --------------------------------------------------------------------------

def test_position_embedding_shape_and_factorization():
    pos = get_3d_sincos_pos_embed(768, (3, 14, 14), add_cls_token=True)
    assert pos.shape == (3 * 14 * 14 + 1, 768)
    assert np.all(pos[0] == 0.0)  # cls row is zeros


def test_encoder_builds_and_runs():
    encoder = PrithviEncoder(img_size=224, patch_size=16, num_frames=3, tubelet_size=1,
                              in_chans=6, embed_dim=64, depth=2, num_heads=4)
    x = np.random.randn(2, 3, 224, 224, 6).astype("float32")
    out = encoder(x)
    num_patches = 3 * (224 // 16) ** 2
    assert tuple(out.shape) == (2, num_patches + 1, 64)


def test_classifier_and_segmenter():
    x = np.random.randn(1, 3, 224, 224, 6).astype("float32")
    clf = PrithviClassifier(variant="prithvi_100m", num_classes=5)
    y = keras.ops.convert_to_numpy(clf(x))
    assert y.shape == (1, 5)

    seg = PrithviSegmenter(variant="prithvi_100m", num_classes=3)
    m = keras.ops.convert_to_numpy(seg(x))
    assert m.shape == (1, 224, 224, 3)


# --------------------------------------------------------------------------
# PyTorch weight-port round-trip against the official architecture
# (timm-style ViT blocks + `get_3d_sincos_pos_embed`), matching the
# convention `build_vit_mapper` assumes.
# --------------------------------------------------------------------------

def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Attn(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(out)

    class Mlp(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.attn = Attn(dim, num_heads)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.mlp = Mlp(dim, int(dim * mlp_ratio))

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class TorchPrithviViT(nn.Module):
        def __init__(self, img_size, patch_size, num_frames, in_chans, embed_dim, depth, num_heads):
            super().__init__()
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size, patch_size),
                                               stride=(1, patch_size, patch_size))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            grid = img_size // patch_size
            pos = get_3d_sincos_pos_embed(embed_dim, (num_frames, grid, grid), add_cls_token=True)
            self.register_buffer("pos_embed", torch.from_numpy(pos)[None])
            self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        def forward(self, x):
            # x: (B, C, T, H, W)
            x = self.patch_embed.proj(x)
            B = x.shape[0]
            x = x.flatten(2).transpose(1, 2)
            x = x + self.pos_embed[:, 1:, :]
            cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

    torch.manual_seed(0)
    img_size, patch_size, num_frames, in_chans, embed_dim, depth, num_heads = 64, 16, 3, 6, 32, 2, 4
    torch_model = TorchPrithviViT(img_size, patch_size, num_frames, in_chans, embed_dim, depth, num_heads)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = PrithviEncoder(img_size=img_size, patch_size=patch_size, num_frames=num_frames,
                                  tubelet_size=1, in_chans=in_chans, embed_dim=embed_dim, depth=depth,
                                  num_heads=num_heads, name="encoder")
    grid = img_size // patch_size
    keras_model(np.zeros((1, num_frames, img_size, img_size, in_chans), dtype="float32"))

    mapper = build_vit_mapper("encoder")
    report = WeightConverter(
        keras_model, state_dict, mapper,
        param_kind_map={"encoder/patch_embed/proj/kernel": "conv3d_kernel"},
    ).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(1, in_chans, num_frames, img_size, img_size).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 4, 1))  # -> (B, T, H, W, C)
    keras_out = keras.ops.convert_to_numpy(keras_model(keras_in, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-3, f"Prithvi weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_prithvi_eo_100m_checkpoint():
    """Downloads the real `ibm-nasa-geospatial/Prithvi-EO-1.0-100M`
    checkpoint and confirms the encoder loads cleanly. Run explicitly with
    `pytest -m pretrained` (network + ~450MB download, cached after the
    first run)."""
    pytest.importorskip("torch")
    from ..weights.pretrained import prithvi_eo_100m

    encoder, report = prithvi_eo_100m()
    assert not report["missing_in_source"]

    x = np.random.rand(1, 3, 224, 224, 6).astype("float32")
    out = keras.ops.convert_to_numpy(encoder(x, training=False))
    num_patches = 3 * (224 // 16) ** 2
    assert out.shape == (1, num_patches + 1, 768)
    assert np.isfinite(out).all()
