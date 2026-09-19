import numpy as np
import pytest
import keras

from keras_climate.foundation.anysat_release import (
    AnySatRelease,
    anysat_projector_configs,
    ANYSAT_MODALITIES,
    ANYSAT_CONFIGS,
    pos_embed_with_scale,
    pos_embed_with_resolution,
    _bucket_ids_2d,
)
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_anysat_release_mapper


def test_all_modalities_have_projector_configs():
    cfgs = anysat_projector_configs(768)
    assert set(cfgs) == set(ANYSAT_MODALITIES)
    for m, cfg in cfgs.items():
        assert cfg["kind"] in ("image", "ts")


def test_pos_embed_shapes():
    pe = pos_embed_with_scale(32, grid_size=4, scale=2, cls_token=True)
    assert pe.shape == (4 * 4 + 1, 32)
    assert np.all(pe[0] == 0.0)

    pe_modis = pos_embed_with_scale(
        32, grid_size=4, scale=2, cls_token=True, modis=True
    )
    assert pe_modis.shape == (4 * 4 + 2, 32)

    pe_res, grid_aug = pos_embed_with_resolution(32, grid_size=2, res=10.0)
    assert pe_res.shape == (grid_aug * grid_aug + 1, 32)


def test_bucket_ids_are_small_and_symmetric_in_count():
    bucket_ids, num_buckets = _bucket_ids_2d(5, 5, skip=1)
    assert bucket_ids.shape == (26, 26)
    assert num_buckets == 8
    assert bucket_ids.max() < num_buckets


def test_single_image_modality_builds_and_runs():
    model = AnySatRelease(
        ["aerial"],
        {"aerial": (100, 100, 4)},
        scale=1,
        embed_dim=32,
        depth=1,
        num_heads=4,
        output="patch",
    )
    out = model({"aerial": np.random.randn(2, 100, 100, 4).astype("float32")})
    assert tuple(out.shape) == (2, 2, 2, 32)


def test_tile_output_mode():
    model = AnySatRelease(
        ["aerial"],
        {"aerial": (50, 50, 4)},
        scale=1,
        embed_dim=32,
        depth=1,
        num_heads=4,
        output="tile",
    )
    out = model({"aerial": np.random.randn(1, 50, 50, 4).astype("float32")})
    assert tuple(out.shape) == (1, 32)


def test_time_series_modality_builds_and_runs():
    model = AnySatRelease(
        ["s2"], {"s2": (5, 2, 2, 10)}, scale=2, embed_dim=32, depth=1, num_heads=4
    )
    out = model(
        {
            "s2": (
                np.random.randn(2, 5, 2, 2, 10).astype("float32"),
                np.random.rand(2, 5).astype("float32") * 300,
            )
        }
    )
    assert tuple(out.shape) == (2, 1, 1, 32)


def test_mixed_image_and_time_series_modalities():
    model = AnySatRelease(
        ["aerial", "s2"],
        {"aerial": (100, 100, 4), "s2": (5, 2, 2, 10)},
        scale=1,
        embed_dim=32,
        depth=1,
        num_heads=4,
    )
    out = model(
        {
            "aerial": np.random.randn(2, 100, 100, 4).astype("float32"),
            "s2": (
                np.random.randn(2, 5, 2, 2, 10).astype("float32"),
                np.random.rand(2, 5).astype("float32") * 300,
            ),
        }
    )
    assert tuple(out.shape) == (2, 2, 2, 32)


def test_inconsistent_footprint_raises():
    with pytest.raises(ValueError, match="patch grid"):
        AnySatRelease(
            ["aerial", "s2"],
            {"aerial": (100, 100, 4), "s2": (5, 4, 4, 10)},
            scale=1,
            embed_dim=32,
            depth=1,
            num_heads=4,
        )


def test_configs_have_expected_base_size():
    assert ANYSAT_CONFIGS["base"] == dict(embed_dim=768, depth=6, num_heads=12)


def test_weight_port_roundtrip_matches_pytorch_reference():
    import math

    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    def unfold(x, dim, size):
        shape = x.shape
        n = shape[dim] // size
        x = x.reshape(shape[:dim] + (n, size) + shape[dim + 1 :])
        return x.movedim(dim + 1, -1)

    class ImageProjector(nn.Module):
        def __init__(self, dim, patch_size, in_chans, resolution):
            super().__init__()
            self.patch_size, self.grid_size = (
                patch_size,
                int(10 / resolution) // patch_size,
            )
            self.patch_embed = nn.Conv2d(
                in_chans, dim, patch_size, patch_size, bias=False
            )
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            )

        def forward(self, x, scale):
            t = self.patch_embed(x)
            gs = self.grid_size
            t = unfold(t, 2, gs)
            t = unfold(t, 3, gs)
            B, D, Ht, Wt = t.shape[:4]
            t = t.reshape(B, D, Ht, Wt, gs * gs)
            t = unfold(t, 2, scale)
            t = unfold(t, 3, scale)
            Htp, Wtp = Ht // scale, Wt // scale
            t = t.reshape(B, D, Htp * Wtp, gs * gs, scale, scale)
            t = t.permute(0, 1, 2, 4, 5, 3).reshape(
                B, D, Htp * Wtp, gs * gs * scale * scale
            )
            t = t.permute(0, 2, 3, 1).reshape(B * Htp * Wtp, gs * gs * scale * scale, D)
            return self.mlp(t)

    class LTAE2dCore(nn.Module):

        def __init__(self, dim, in_channels, n_head, d_k, T):
            super().__init__()
            mlp_in = [dim // 8, dim // 2, dim, dim * 2, dim]
            self.n_head, self.d_k, self.T = n_head, d_k, T
            self.d_model = mlp_in[-1]
            dims = [in_channels] + mlp_in
            self.inconv = nn.Sequential(
                *sum(
                    (
                        [
                            nn.Linear(dims[i], dims[i + 1]),
                            nn.GroupNorm(4, dims[i + 1]),
                            nn.ReLU(),
                            nn.Dropout(0.0),
                        ]
                        for i in range(len(dims) - 1)
                    ),
                    [],
                )
            )
            self.in_norm = nn.GroupNorm(n_head, self.d_model)
            self.attention_heads = nn.Module()
            self.attention_heads.fc1_k = nn.Linear(self.d_model, n_head * d_k)
            self.attention_heads.Q = nn.Parameter(torch.zeros(n_head, d_k))
            self.mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GroupNorm(4, self.d_model),
                nn.ReLU(),
            )
            self.out_norm = nn.GroupNorm(n_head, self.d_model)

        def _pos_enc(self, bp):
            d = self.d_model // self.n_head
            denom = self.T ** (2 * (torch.arange(d).float() // 2) / d)
            table = bp[:, :, None] / denom[None, None, :]
            out = torch.empty_like(table)
            out[:, :, 0::2] = table[:, :, 0::2].sin()
            out[:, :, 1::2] = table[:, :, 1::2].cos()
            return out.repeat(1, 1, self.n_head)

        def forward(self, x, dates):
            B, T, Cin, H, W = x.shape
            out = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, Cin)
            out = self.inconv(out.reshape(-1, Cin)).reshape(B * H * W, T, -1)
            out = self.in_norm(out.transpose(1, 2)).transpose(1, 2)
            bp = dates[:, None, None, :].expand(-1, H, W, -1).permute(0, 3, 1, 2)
            bp = bp.permute(0, 2, 3, 1).reshape(B * H * W, T)
            out = out + self._pos_enc(bp)

            N = B * H * W
            Q, fc1_k = self.attention_heads.Q, self.attention_heads.fc1_k
            q = Q[:, None, :].expand(-1, N, -1).reshape(self.n_head * N, self.d_k)
            k = (
                fc1_k(out)
                .reshape(N, T, self.n_head, self.d_k)
                .permute(2, 0, 1, 3)
                .reshape(self.n_head * N, T, self.d_k)
            )
            v = (
                out.reshape(N, T, self.n_head, -1)
                .permute(2, 0, 1, 3)
                .reshape(self.n_head * N, T, -1)
            )
            attn = F.softmax(
                (q[:, None, :] @ k.transpose(1, 2)) / self.d_k**0.5, dim=-1
            )
            o = (attn @ v).reshape(self.n_head, N, -1).permute(1, 0, 2).reshape(N, -1)
            o = self.out_norm(self.mlp(o))
            return o.reshape(B, H, W, -1).permute(0, 3, 1, 2)

    class TSProjector(nn.Module):
        def __init__(self, dim, in_channels, n_head=4, d_k=3, T=100, reduce_scale=1):
            super().__init__()
            self.patch_embed = LTAE2dCore(dim, in_channels, n_head, d_k, T)
            self.reduce_scale = reduce_scale

        def forward(self, x, dates, scale):
            B, T, Cin, H, W = x.shape
            o = self.patch_embed(x, dates)

            se = max(1, scale // self.reduce_scale)
            o = unfold(o, 2, se)
            o = unfold(o, 3, se)
            Hp, Wp = H // se, W // se
            o = (
                o.reshape(B, -1, Hp * Wp, se * se)
                .permute(0, 2, 3, 1)
                .reshape(B * Hp * Wp, se * se, -1)
            )
            return o

    def sincos_scale(dim, grid_size, scale, cls_token=True):
        gh, gw = torch.arange(grid_size).float(), torch.arange(grid_size).float()
        Y, X = torch.meshgrid(gh, gw, indexing="ij")
        grid = torch.stack([X, Y], 0) * scale
        omega = torch.arange(dim // 4).float() / (dim // 4)
        omega = 1.0 / (10000**omega)

        def emb1d(pos):
            out = torch.einsum("m,d->md", pos.reshape(-1), omega)
            return torch.cat([out.sin(), out.cos()], -1)

        pe = torch.cat([emb1d(grid[0]), emb1d(grid[1])], -1)
        if cls_token:
            pe = torch.cat([torch.zeros(1, dim), pe], 0)
        return pe

    def sincos_res(dim, grid_size, res, cls_token=True):
        grid_aug = max(1, int(grid_size * 10 / res))
        pe = sincos_scale(dim, grid_aug, res, cls_token=False)
        if cls_token:
            pe = torch.cat([torch.zeros(1, dim), pe], 0)
        return pe, grid_aug

    def bucket_ids(height, width, skip=1):
        ratio = 1.9
        alpha, beta, gamma = ratio, 2 * ratio, 8 * ratio
        beta_int = int(beta)
        rows = torch.arange(height).view(-1, 1).repeat(1, width)
        cols = torch.arange(width).view(1, -1).repeat(height, 1)
        pos = torch.stack([rows, cols], -1).float()
        L = height * width
        diff = pos.reshape(L, 1, 2) - pos.reshape(1, L, 2)
        dis = diff.square().sum(-1).sqrt().round()
        idx = dis.round()
        mask = dis.abs() > alpha
        inner = alpha + torch.log(dis[mask] / alpha) / math.log(gamma / alpha) * (
            beta - alpha
        )
        idx[mask] = inner.round().clamp(max=beta)
        bucket = (idx + beta_int).long()
        num_buckets = 2 * beta_int + 1
        out = torch.full((skip + L, skip + L), num_buckets, dtype=torch.long)
        out[skip:, skip:] = bucket
        return out, num_buckets + 1

    class RPEBiasK(nn.Module):
        def __init__(self, head_dim, num_buckets):
            super().__init__()
            self.lookup_table_weight = nn.Parameter(
                torch.zeros(1, head_dim, num_buckets)
            )

        def forward(self, q, bucket):
            B, num_buckets = q.shape[0], self.lookup_table_weight.shape[-1]
            lookup = torch.einsum("bld,dn->bln", q[:, 0], self.lookup_table_weight[0])
            Lq, Lk = bucket.shape
            offset = torch.arange(Lq).view(-1, 1) * num_buckets
            flat_idx = (
                (bucket + offset).flatten().unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            )
            bias = torch.gather(lookup.unsqueeze(1).flatten(2), 2, flat_idx)
            return bias.reshape(B, 1, Lq, Lk)

    class Attn(nn.Module):
        def __init__(self, dim, num_heads, rpe=False):
            super().__init__()
            self.num_heads, self.head_dim = num_heads, dim // num_heads
            self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
            self.rpe_k = RPEBiasK(self.head_dim, 8) if rpe else None

        def forward(self, x, bucket=None):
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.head_dim**-0.5
            attn = q @ k.transpose(-2, -1)
            if self.rpe_k is not None:
                attn = attn + self.rpe_k(q, bucket)
            attn = attn.softmax(-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return self.proj(out)

    class Mlp(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(dim, dim * 4), nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x)))

    class Block(nn.Module):
        def __init__(self, dim, num_heads, rpe=False, eps=1e-6):
            super().__init__()
            self.norm1, self.norm2 = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(
                dim, eps=eps
            )
            self.attn = Attn(dim, num_heads, rpe=rpe)
            self.mlp = Mlp(dim)

        def forward(self, x, bucket=None):
            x = x + self.attn(self.norm1(x), bucket)
            x = x + self.mlp(self.norm2(x))
            return x

    class LocalEncoder(nn.Module):
        def __init__(self, dim, depth, num_heads):
            super().__init__()
            self.dim = dim
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.predictor_blocks = nn.ModuleList(
                [Block(dim, num_heads, rpe=True, eps=1e-5) for _ in range(depth)]
            )
            self.predictor_norm = nn.LayerNorm(dim, eps=1e-5)

        def forward(self, tokens, pos_embed, bucket):
            B_ = tokens.shape[0]
            cls = self.cls_token.expand(B_, -1, -1)
            x = torch.cat([cls, tokens], 1) + pos_embed
            for blk in self.predictor_blocks:
                x = blk(x, bucket)
            return self.predictor_norm(x)[:, 0]

    class CrossAttn(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.num_heads, self.head_dim = num_heads, dim // num_heads
            self.wk, self.wv, self.proj = (
                nn.Linear(dim, dim),
                nn.Linear(dim, dim),
                nn.Linear(dim, dim),
            )
            self.q_learned = nn.Parameter(torch.zeros(1, 1, dim))
            self.rpe_k = RPEBiasK(self.head_dim, 8)

        def forward(self, h, pos_embed_q, n_modalities, bucket):
            B, N, C = h.shape
            Nq = pos_embed_q.shape[1]
            q_ = self.q_learned + pos_embed_q
            q = (
                q_.expand(B, -1, -1)
                .reshape(B, Nq, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = self.wk(h).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.wv(h).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
            rpe = self.rpe_k(q, bucket)
            rpe_full = torch.cat(
                [rpe[:, :, :, :1], rpe[:, :, :, 1:].repeat(1, 1, 1, n_modalities)], -1
            )
            attn = (attn + rpe_full).softmax(-1)
            out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
            return self.proj(out)

    class CrossPool(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(
                dim, eps=1e-6
            )
            self.attn = CrossAttn(dim, num_heads)
            self.mlp = Mlp(dim)

        def forward(self, x, pos_embed_q, n_modalities, bucket):
            pooled = self.attn(self.norm1(x), pos_embed_q, n_modalities, bucket)
            return pooled + self.mlp(self.norm2(pooled))

    class TorchAnySatRelease(nn.Module):
        def __init__(self, dim, depth, num_heads, aerial_cfg, s2_cfg, input_res):
            super().__init__()
            self.dim = dim
            self.projector_aerial = ImageProjector(dim, **aerial_cfg)
            self.projector_s2 = TSProjector(dim, **s2_cfg)
            self.spatial_encoder = LocalEncoder(dim, depth, num_heads)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.blocks = nn.ModuleList(
                [Block(dim, num_heads, rpe=False) for _ in range(depth)]
                + [CrossPool(dim, num_heads)]
            )
            self.input_res = input_res

        def forward(self, x_aerial, x_s2, dates_s2, scale):
            aerial_tok = self.projector_aerial(x_aerial, scale)
            s2_tok = self.projector_s2(x_s2, dates_s2, scale)

            pe_aerial, ga_aerial = sincos_res(self.dim, scale, self.input_res["aerial"])
            pe_s2, ga_s2 = sincos_res(self.dim, scale, self.input_res["s2"])
            bucket_aerial, _ = bucket_ids(ga_aerial, ga_aerial, skip=1)
            bucket_s2, _ = bucket_ids(ga_s2, ga_s2, skip=1)

            aerial_pooled = self.spatial_encoder(aerial_tok, pe_aerial, bucket_aerial)
            s2_pooled = self.spatial_encoder(s2_tok, pe_s2, bucket_s2)

            B = x_aerial.shape[0]
            num_patches_side = int(round((aerial_pooled.shape[0] / B) ** 0.5))
            num_patches = num_patches_side * num_patches_side
            aerial_pooled = aerial_pooled.reshape(B, num_patches, self.dim)
            s2_pooled = s2_pooled.reshape(B, num_patches, self.dim)

            pe_global = sincos_scale(self.dim, num_patches_side, scale, cls_token=True)
            cls = (self.cls_token + pe_global[:1]).expand(B, -1, -1)
            tokens = torch.cat(
                [cls, aerial_pooled + pe_global[1:], s2_pooled + pe_global[1:]], 1
            )

            for blk in self.blocks[:-1]:
                tokens = blk(tokens)

            bucket_cross, _ = bucket_ids(num_patches_side, num_patches_side, skip=1)
            tokens = self.blocks[-1](
                tokens, pe_global.unsqueeze(0), n_modalities=2, bucket=bucket_cross
            )
            return tokens[:, 1:].reshape(
                B, num_patches_side, num_patches_side, self.dim
            )

    torch.manual_seed(0)
    np.random.seed(0)

    dim, num_heads, depth, scale = 32, 4, 2, 2
    aerial_cfg = dict(patch_size=10, in_chans=4, resolution=0.2)
    s2_cfg = dict(in_channels=10, n_head=16, d_k=8, T=100, reduce_scale=1)
    input_res = {"aerial": 2, "s2": 10}

    torch_model = TorchAnySatRelease(
        dim, depth, num_heads, aerial_cfg, s2_cfg, input_res
    )
    torch_model.eval()
    for p in torch_model.parameters():
        p.data.normal_(0, 0.3)

    B = 2
    img = 5 * 4 * aerial_cfg["patch_size"]
    x_aerial = torch.randn(B, 4, img, img)
    T = 5
    x_s2 = torch.randn(B, T, 10, 4, 4)
    dates_s2 = torch.rand(B, T) * 100

    with torch.no_grad():
        out_t = torch_model(x_aerial, x_s2, dates_s2, scale).numpy()

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}
    keras_model = AnySatRelease(
        ["aerial", "s2"],
        {"aerial": (img, img, 4), "s2": (T, 4, 4, 10)},
        scale,
        embed_dim=dim,
        depth=depth,
        num_heads=num_heads,
    )
    keras_model(
        {
            "aerial": np.zeros((B, img, img, 4), dtype="float32"),
            "s2": (
                np.zeros((B, T, 4, 4, 10), dtype="float32"),
                np.zeros((B, T), dtype="float32"),
            ),
        }
    )

    mapper = build_anysat_release_mapper(["aerial", "s2"], depth)
    report = WeightConverter(
        keras_model, state_dict, mapper, skip_patterns=[r"pad_parameter$"]
    ).convert(strict=False, verbose=False)
    assert not report["unused_source_keys"]
    assert report["missing_in_source"] == ["anysat_release/projector_s2/pe_denom"]

    x_aerial_np = x_aerial.numpy().transpose(0, 2, 3, 1).astype("float32")
    x_s2_np = x_s2.numpy().transpose(0, 1, 3, 4, 2).astype("float32")
    out_k = keras.ops.convert_to_numpy(
        keras_model(
            {"aerial": x_aerial_np, "s2": (x_s2_np, dates_s2.numpy().astype("float32"))}
        )
    )

    max_diff = np.abs(out_t - out_k).max()
    assert (
        max_diff < 1e-3
    ), f"AnySatRelease weight port numerical mismatch: max abs diff {max_diff}"
