import numpy as np
import pytest
import keras

from keras_climate.remote_sensing.satmae import SatMAE, SatMAEEncoder, SatMAEDecoder
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_satmae_mapper
from keras_climate.weights.pretrained import satmae_vit_base_mae


def test_single_mode_mae_pretraining_forward():
    model = SatMAE(img_size=64, patch_size=16, in_chans=3, embed_dim=32, depth=2, num_heads=4,
                   decoder_embed_dim=24, decoder_depth=2, decoder_num_heads=4, mode="single")
    x = np.random.randn(2, 64, 64, 3).astype("float32")
    pred, mask = model(x)
    num_patches = (64 // 16) ** 2
    assert tuple(pred.shape) == (2, num_patches, 16 * 16 * 3)
    assert tuple(mask.shape) == (2, num_patches)


@pytest.mark.parametrize("mode,kwarg", [("multispectral", "num_groups"), ("temporal", "num_frames")])
def test_multi_group_modes_forward(mode, kwarg):
    model = SatMAE(img_size=64, patch_size=16, in_chans=3, embed_dim=32, depth=2, num_heads=4,
                   decoder_embed_dim=24, decoder_depth=2, decoder_num_heads=4,
                   mode=mode, **{kwarg: 3})
    x = np.random.randn(2, 3, 64, 64, 3).astype("float32")
    pred, mask = model(x)
    num_patches = (64 // 16) ** 2 * 3
    assert tuple(pred.shape) == (2, num_patches, 16 * 16 * 3)
    assert tuple(mask.shape) == (2, num_patches)


def test_encoder_alone_for_downstream_tasks():
    encoder = SatMAEEncoder(img_size=64, patch_size=16, in_chans=3, embed_dim=32, depth=2, num_heads=4)
    x = np.random.randn(2, 64, 64, 3).astype("float32")
    tokens = encoder(x)
    num_patches = (64 // 16) ** 2
    assert tuple(tokens.shape) == (2, num_patches + 1, 32)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    def sincos_pos_embed(grid_size, dim):
        def embed_1d(pos, d):
            omega = np.arange(d // 2, dtype=np.float32) / (d / 2.0)
            omega = 1.0 / (10000 ** omega)
            pos = pos.reshape(-1)
            out = np.einsum("m,d->md", pos, omega)
            return np.concatenate([np.sin(out), np.cos(out)], axis=1)

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
        emb_h = embed_1d(grid[0], dim // 2)
        emb_w = embed_1d(grid[1], dim // 2)
        return np.concatenate([emb_h, emb_w], axis=1)

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

    class ViTBlock(nn.Module):
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

    class TorchMAE(nn.Module):
        def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                     decoder_embed_dim, decoder_depth, decoder_num_heads):
            super().__init__()
            grid = img_size // patch_size
            num_patches = grid * grid

            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                               stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos = sincos_pos_embed(grid, embed_dim)
            pos = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos], axis=0)
            self.pos_embed = nn.Parameter(torch.from_numpy(pos)[None], requires_grad=False)
            self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            dpos = sincos_pos_embed(grid, decoder_embed_dim)
            dpos = np.concatenate([np.zeros((1, decoder_embed_dim), dtype=np.float32), dpos], axis=0)
            self.decoder_pos_embed = nn.Parameter(torch.from_numpy(dpos)[None], requires_grad=False)
            self.decoder_blocks = nn.ModuleList(
                [ViTBlock(decoder_embed_dim, decoder_num_heads) for _ in range(decoder_depth)])
            self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans,
                                           bias=True)

        def forward_encoder(self, x):
            x = self.patch_embed.proj(x)
            B = x.shape[0]
            x = x.flatten(2).transpose(1, 2)
            x = x + self.pos_embed[:, 1:, :]
            cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)

        def forward_decoder(self, x):
            x = self.decoder_embed(x)
            x = x + self.decoder_pos_embed
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
            x = self.decoder_pred(x)
            return x[:, 1:, :]

        def forward(self, x):
            return self.forward_decoder(self.forward_encoder(x))

    torch.manual_seed(0)
    img, patch, in_chans, embed, depth, heads = 64, 16, 3, 32, 2, 4
    dec_embed, dec_depth, dec_heads = 24, 2, 4

    torch_model = TorchMAE(img, patch, in_chans, embed, depth, heads, dec_embed, dec_depth, dec_heads)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    encoder = SatMAEEncoder(img_size=img, patch_size=patch, in_chans=in_chans, embed_dim=embed,
                             depth=depth, num_heads=heads, mode="single")
    decoder = SatMAEDecoder(num_patches=(img // patch) ** 2, patch_size=patch, in_chans=in_chans,
                             decoder_embed_dim=dec_embed, decoder_depth=dec_depth,
                             decoder_num_heads=dec_heads, encoder_embed_dim=embed)

    x0 = np.zeros((1, img, img, in_chans), dtype="float32")
    tok0, mask0, ids0 = encoder(x0, apply_masking=True, mask_ratio=0.0)
    decoder(tok0, ids0)

    mapper = build_satmae_mapper()
    enc_report = WeightConverter(encoder, state_dict, mapper).convert(strict=False, verbose=False)
    dec_report = WeightConverter(decoder, state_dict, mapper).convert(strict=False, verbose=False)
    assert not enc_report["missing_in_source"]
    assert not dec_report["missing_in_source"]

    x_np = np.random.randn(1, in_chans, img, img).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    tokens, mask, ids_restore = encoder(keras_in, apply_masking=True, mask_ratio=0.0)
    keras_out = keras.ops.convert_to_numpy(decoder(tokens, ids_restore))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"SatMAE weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_mae_vit_base_checkpoint():
    pytest.importorskip("torch")
    encoder, decoder, reports = satmae_vit_base_mae(img_size=224)
    assert not reports["encoder"]["missing_in_source"]
    assert not reports["decoder"]["missing_in_source"]

    x = np.random.rand(1, 224, 224, 3).astype("float32")
    tokens, mask, ids_restore = encoder(x, apply_masking=True, mask_ratio=0.75)
    pred = keras.ops.convert_to_numpy(decoder(tokens, ids_restore))
    assert pred.shape == (1, (224 // 16) ** 2, 16 * 16 * 3)
    assert np.isfinite(pred).all()
