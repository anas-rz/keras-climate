import numpy as np
import pytest
import keras

from keras_climate.foundation.clay import (
    ClayEncoder,
    ClayClassifier,
    posemb_sincos_2d_with_gsd,
    posemb_sincos_1d,
)
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import (
    convert_clay_encoder_state_dict,
    build_clay_identity_mapper,
)


def test_position_embedding_shapes():
    pos = posemb_sincos_2d_with_gsd(8, 8, 56, gsd=10.0)
    assert pos.shape == (64, 56)
    waves = posemb_sincos_1d(np.array([0.49, 0.56, 0.665], dtype="float32"), 32)
    assert waves.shape == (3, 32)


def test_encoder_builds_and_runs():
    encoder = ClayEncoder(
        img_size=64,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
        dim_head=16,
        wave_dim=32,
        num_latent_tokens=8,
    )
    pixels = np.random.randn(2, 64, 64, 6).astype("float32")
    waves = np.array([0.49, 0.56, 0.665, 0.705, 0.74, 0.783], dtype="float32")
    time_latlon = np.random.randn(2, 8).astype("float32")
    out = encoder(
        {"pixels": pixels, "waves": waves, "time_latlon": time_latlon}, gsd=10.0
    )
    num_patches = (64 // 8) ** 2
    assert tuple(out.shape) == (2, num_patches + 1, 64)


def test_classifier():
    pixels = np.random.randn(1, 64, 64, 6).astype("float32")
    waves = np.array([0.49, 0.56, 0.665, 0.705, 0.74, 0.783], dtype="float32")
    time_latlon = np.random.randn(1, 8).astype("float32")
    clf = ClayClassifier(
        img_size=64,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        num_bands=6,
        num_classes=5,
    )
    y = keras.ops.convert_to_numpy(clf([pixels, time_latlon, waves[None]]))
    assert y.shape == (1, 5)


def test_different_band_counts_reuse_the_same_dynamic_embedding():
    encoder = ClayEncoder(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=1,
        num_heads=4,
        dim_head=8,
        wave_dim=16,
        num_latent_tokens=4,
    )
    for num_bands in (3, 6, 10):
        pixels = np.random.randn(1, 32, 32, num_bands).astype("float32")
        waves = np.linspace(0.4, 2.2, num_bands).astype("float32")
        time_latlon = np.random.randn(1, 8).astype("float32")
        out = encoder(
            {"pixels": pixels, "waves": waves, "time_latlon": time_latlon}, gsd=10.0
        )
        assert out.shape[-1] == 32


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    def torch_posemb_sincos_1d(waves, dim, temperature=10000.0):
        omega = torch.arange(dim // 2) / (dim // 2 - 1)
        omega = 1.0 / (temperature**omega)
        scaled = waves[:, None] * omega[None, :]
        return torch.cat([scaled.sin(), scaled.cos()], dim=1)

    def torch_posemb_sincos_2d_with_gsd(h, w, dim, gsd=1.0, temperature=10000.0):
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature ** (2 * omega / dim)) * gsd
        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1).float()

    class FCBlock(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.l1 = nn.Linear(size, size)
            self.l2 = nn.Linear(size, size)

        def forward(self, x):
            y = F.gelu(self.l1(x))
            y = F.gelu(self.l2(y))
            return x + y

    class WavesTransformer(nn.Module):
        def __init__(
            self,
            wave_dim,
            output_dim,
            num_latent_tokens,
            embed_dim,
            num_heads=4,
            num_layers=1,
        ):
            super().__init__()
            self.num_latent_tokens = num_latent_tokens
            layer = nn.TransformerEncoderLayer(
                d_model=wave_dim,
                nhead=num_heads,
                activation="gelu",
                dropout=0,
                norm_first=False,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            self.fc_weight = nn.Linear(wave_dim, output_dim)
            self.fc_bias = nn.Linear(wave_dim, embed_dim)
            self.weight_tokens = nn.Parameter(
                torch.randn(num_latent_tokens, wave_dim) * 0.02
            )
            self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

        def forward(self, x):
            x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
            out = self.encoder(x)
            weights = self.fc_weight(
                out[self.num_latent_tokens : -1] + x[self.num_latent_tokens : -1]
            )
            bias = self.fc_bias(out[-1])
            return weights, bias

    class DynamicEmbedding(nn.Module):
        def __init__(self, wave_dim, num_latent_tokens, patch_size, embed_dim):
            super().__init__()
            self.wave_dim = wave_dim
            self.patch_size = patch_size
            self.embed_dim = embed_dim
            self.weight_generator = WavesTransformer(
                wave_dim,
                patch_size * patch_size * embed_dim,
                num_latent_tokens,
                embed_dim,
            )
            self.fclayer = FCBlock(wave_dim)

        def forward(self, batch, waves):
            waves_enc = torch_posemb_sincos_1d(waves, self.wave_dim)
            waves_enc = self.fclayer(waves_enc)
            weight, bias = self.weight_generator(waves_enc)
            dynamic_weight = weight.reshape(
                -1, self.embed_dim, self.patch_size, self.patch_size
            ).permute(1, 0, 2, 3)
            out = F.conv2d(
                batch, dynamic_weight * 0.02, bias=bias * 0.02, stride=self.patch_size
            )
            return out.flatten(2).transpose(1, 2), waves_enc

    class Attention(nn.Module):
        def __init__(self, dim, heads, dim_head):
            super().__init__()
            inner_dim = dim_head * heads
            self.heads, self.dim_head = heads, dim_head
            self.scale = dim_head**-0.5
            self.norm = nn.LayerNorm(dim)
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)

        def forward(self, x):
            x = self.norm(x)
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            B, N, _ = x.shape
            q, k, v = [
                t.reshape(B, N, self.heads, self.dim_head).transpose(1, 2) for t in qkv
            ]
            attn = ((q @ k.transpose(-1, -2)) * self.scale).softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.heads * self.dim_head)
            return self.to_out(x)

    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, x):
            return self.net(x)

    class Transformer(nn.Module):
        def __init__(self, dim, depth, heads, dim_head, mlp_dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [Attention(dim, heads, dim_head), FeedForward(dim, mlp_dim)]
                    )
                    for _ in range(depth)
                ]
            )

        def forward(self, x):
            for attn, ff in self.layers:
                x = attn(x) + x
                x = ff(x) + x
            return self.norm(x)

    class TorchClayEncoder(nn.Module):
        def __init__(
            self,
            img_size,
            patch_size,
            embed_dim,
            depth,
            num_heads,
            dim_head,
            mlp_ratio,
            wave_dim,
            num_latent_tokens,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.patch_size = patch_size
            self.grid = img_size // patch_size
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.patch_embedding = DynamicEmbedding(
                wave_dim, num_latent_tokens, patch_size, embed_dim
            )
            self.transformer = Transformer(
                embed_dim, depth, num_heads, dim_head, int(embed_dim * mlp_ratio)
            )

        def forward(self, pixels, waves, time_latlon, gsd=1.0):
            patches, _ = self.patch_embedding(pixels, waves)
            pos = torch_posemb_sincos_2d_with_gsd(
                self.grid, self.grid, self.embed_dim - 8, gsd=gsd
            ).to(patches)
            B, L, _ = patches.shape
            pos = pos.unsqueeze(0).expand(B, -1, -1)
            tl = time_latlon.unsqueeze(1).expand(-1, L, -1)
            patches = patches + torch.cat([pos, tl], dim=-1)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, patches], dim=1)
            return self.transformer(tokens)

    IMG, PATCH, EMBED, DEPTH, HEADS, DIM_HEAD = 64, 8, 64, 2, 4, 16
    WAVE_DIM, NUM_LATENT, NUM_BANDS = 32, 8, 6

    torch.manual_seed(0)
    torch_model = TorchClayEncoder(
        IMG, PATCH, EMBED, DEPTH, HEADS, DIM_HEAD, 4.0, WAVE_DIM, NUM_LATENT
    )
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    flat = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}
    translated = convert_clay_encoder_state_dict(
        flat, depth=DEPTH, num_latent_tokens=NUM_LATENT
    )

    keras_model = ClayEncoder(
        img_size=IMG,
        patch_size=PATCH,
        embed_dim=EMBED,
        depth=DEPTH,
        num_heads=HEADS,
        dim_head=DIM_HEAD,
        wave_dim=WAVE_DIM,
        num_latent_tokens=NUM_LATENT,
    )
    zeros = {
        "pixels": np.zeros((1, IMG, IMG, NUM_BANDS), dtype="float32"),
        "waves": np.zeros((NUM_BANDS,), dtype="float32"),
        "time_latlon": np.zeros((1, 8), dtype="float32"),
    }
    keras_model(zeros, gsd=10.0)

    report = WeightConverter(
        keras_model, translated, build_clay_identity_mapper()
    ).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    waves = np.array([0.49, 0.56, 0.665, 0.705, 0.74, 0.783], dtype="float32")
    pixels = np.random.randn(2, IMG, IMG, NUM_BANDS).astype("float32")
    time_latlon = np.random.randn(2, 8).astype("float32")
    gsd = 10.0

    with torch.no_grad():
        torch_pixels = torch.from_numpy(np.transpose(pixels, (0, 3, 1, 2)))
        torch_out = torch_model(
            torch_pixels,
            torch.from_numpy(waves),
            torch.from_numpy(time_latlon),
            gsd=gsd,
        ).numpy()

    keras_out = keras.ops.convert_to_numpy(
        keras_model(
            {"pixels": pixels, "waves": waves, "time_latlon": time_latlon},
            gsd=gsd,
            training=False,
        )
    )

    max_diff = np.abs(torch_out - keras_out).max()
    assert (
        max_diff < 1e-3
    ), f"Clay weight port numerical mismatch: max abs diff {max_diff}"
