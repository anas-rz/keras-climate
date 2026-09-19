import numpy as np
import keras
from keras import layers, ops

ANYSAT_CONFIGS = {
    "tiny": dict(embed_dim=256, depth=2, num_heads=4),
    "small": dict(embed_dim=512, depth=4, num_heads=8),
    "base": dict(embed_dim=768, depth=6, num_heads=12),
}

ANYSAT_MODALITIES = [
    "aerial",
    "aerial-flair",
    "spot",
    "naip",
    "s2",
    "s1-asc",
    "s1",
    "alos",
    "l7",
    "l8",
    "modis",
]

ANYSAT_IMAGE_MODALITIES = {"aerial", "aerial-flair", "spot", "naip"}

ANYSAT_INPUT_RES = {
    "aerial": 2,
    "aerial-flair": 2,
    "spot": 10,
    "naip": 10,
    "s2": 10,
    "s1-asc": 10,
    "s1-des": 10,
    "s1": 10,
    "l8": 10,
    "l7": 30,
    "alos": 30,
    "modis": 250,
}

_TS_LTAE_HEAD = dict(n_head=16, d_k=8)


def anysat_projector_configs(embed_dim):
    dim = embed_dim
    mlp_in = [dim // 8, dim // 2, dim, dim * 2, dim]
    configs = {
        "aerial": dict(kind="image", patch_size=10, in_chans=4, resolution=0.2),
        "aerial-flair": dict(kind="image", patch_size=10, in_chans=5, resolution=0.2),
        "spot": dict(kind="image", patch_size=10, in_chans=3, resolution=1.0),
        "naip": dict(kind="image", patch_size=8, in_chans=4, resolution=1.25),
        "s2": dict(
            kind="ts",
            in_channels=10,
            T=367,
            in_norm=True,
            reduce_scale=1,
            mlp_in=list(mlp_in),
        ),
        "s1-asc": dict(
            kind="ts",
            in_channels=2,
            T=367,
            in_norm=False,
            reduce_scale=1,
            mlp_in=list(mlp_in),
        ),
        "s1": dict(
            kind="ts",
            in_channels=3,
            T=367,
            in_norm=False,
            reduce_scale=1,
            mlp_in=list(mlp_in),
        ),
        "alos": dict(
            kind="ts",
            in_channels=3,
            T=367,
            in_norm=False,
            reduce_scale=1,
            mlp_in=list(mlp_in),
        ),
        "l7": dict(
            kind="ts",
            in_channels=6,
            T=367,
            in_norm=False,
            reduce_scale=1,
            mlp_in=list(mlp_in),
        ),
        "l8": dict(
            kind="ts",
            in_channels=11,
            T=366,
            in_norm=False,
            reduce_scale=1,
            mlp_in=list(mlp_in),
        ),
        "modis": dict(
            kind="ts",
            in_channels=7,
            T=367,
            in_norm=False,
            reduce_scale=12,
            mlp_in=list(mlp_in),
        ),
    }
    return configs


def _sincos_1d(dim_half, pos):
    omega = np.arange(dim_half // 2, dtype=np.float64) / (dim_half / 2.0)
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1).astype(np.float64)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _sincos_2d_from_grid(dim, grid):
    emb_h = _sincos_1d(dim // 2, grid[0])
    emb_w = _sincos_1d(dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def pos_embed_with_scale(dim, grid_size, scale, cls_token=True, modis=False):
    gh = np.arange(grid_size, dtype=np.float64)
    gw = np.arange(grid_size, dtype=np.float64)
    X, Y = np.meshgrid(gw, gh)
    grid = np.stack([X, Y], axis=0)[:, None] * scale
    pe = _sincos_2d_from_grid(dim, grid).reshape(grid_size * grid_size, dim)
    if cls_token:
        pe = np.concatenate([np.zeros((1, dim)), pe], axis=0)
    if modis:
        pe = np.concatenate([np.zeros((1, dim)), pe], axis=0)
    return pe.astype("float32")


def pos_embed_with_resolution(dim, grid_size, res, cls_token=True):
    grid_aug = max(1, int(grid_size * 10 / res))
    gh = np.arange(grid_aug, dtype=np.float64)
    gw = np.arange(grid_aug, dtype=np.float64)
    X, Y = np.meshgrid(gw, gh)
    grid = np.stack([X, Y], axis=0)[:, None] * res
    pe = _sincos_2d_from_grid(dim, grid).reshape(grid_aug * grid_aug, dim)
    if cls_token:
        pe = np.concatenate([np.zeros((1, dim)), pe], axis=0)
    return pe.astype("float32"), grid_aug


_RATIO = 1.9
_ALPHA, _BETA, _GAMMA = 1 * _RATIO, 2 * _RATIO, 8 * _RATIO
_BETA_INT = int(_BETA)
_NUM_BUCKETS_NOSKIP = 2 * _BETA_INT + 1


def _piecewise_index(rel):
    rel = np.asarray(rel, dtype=np.float64)
    idx = np.round(rel)
    rp_abs = np.abs(rel)
    not_mask = rp_abs > _ALPHA
    rp_out = rel[not_mask]
    rp_abs_out = rp_abs[not_mask]
    inner = _ALPHA + np.log(rp_abs_out / _ALPHA) / np.log(_GAMMA / _ALPHA) * (
        _BETA - _ALPHA
    )
    inner = np.minimum(np.round(inner), _BETA)
    idx[not_mask] = np.sign(rp_out) * inner
    return idx.astype(np.int64)


def _bucket_ids_2d(height, width, skip=1):
    rows = np.arange(height).reshape(height, 1).repeat(width, axis=1)
    cols = np.arange(width).reshape(1, width).repeat(height, axis=0)
    pos = np.stack([rows, cols], axis=2)
    L = height * width
    diff = pos.reshape(L, 1, 2) - pos.reshape(1, L, 2)
    dis = np.round(np.sqrt(np.sum(diff.astype(np.float64) ** 2, axis=2)))
    bucket = (_piecewise_index(dis) + _BETA_INT).reshape(L, L)
    num_buckets = _NUM_BUCKETS_NOSKIP
    if skip > 0:
        new_b = np.empty((skip + L, skip + L), dtype=bucket.dtype)
        extra = num_buckets
        num_buckets += 1
        new_b[:skip] = extra
        new_b[:, :skip] = extra
        new_b[skip:, skip:] = bucket
        bucket = new_b
    return bucket, num_buckets


def _bucket_flat_offset_index(height, width, skip=1):
    bucket, num_buckets = _bucket_ids_2d(height, width, skip)
    Lq, Lk = bucket.shape
    offset = (np.arange(Lq) * num_buckets)[:, None]
    return (bucket + offset).reshape(-1).astype("int32"), num_buckets, Lq, Lk


def _unfold(x, dim, size):
    shape = ops.shape(x)
    static_shape = x.shape
    n = static_shape[dim] // size
    new_shape = tuple(shape[:dim]) + (n, size) + tuple(shape[dim + 1 :])
    x = ops.reshape(x, new_shape)
    perm = list(range(len(new_shape)))
    axis = dim + 1
    perm = perm[:axis] + perm[axis + 1 :] + [axis]
    return ops.transpose(x, perm)


def _rpe_bias_k(q, weight, flat_idx, num_buckets, Lq, Lk):
    B = ops.shape(q)[0]
    q0 = q[:, :1]
    lookup = ops.einsum("bhld,dn->bhln", q0, weight[0])
    lookup_flat = ops.reshape(lookup, (B, 1, Lq * num_buckets))
    bias_flat = ops.take(lookup_flat, flat_idx, axis=-1)
    return ops.reshape(bias_flat, (B, 1, Lq, Lk))


class AnySatImageProjector(layers.Layer):

    def __init__(
        self, embed_dim, patch_size, in_chans, resolution, scale, bias=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.resolution = resolution
        self.scale = scale
        self.use_bias = bias
        self.res = int(10 / resolution)
        self.grid_size = self.res // patch_size

    def build(self, input_shape):
        H, W = input_shape[1], input_shape[2]
        ps, gs, scale = self.patch_size, self.grid_size, self.scale
        self.Hc, self.Wc = H // ps, W // ps
        self.Ht, self.Wt = self.Hc // gs, self.Wc // gs
        self.Htp, self.Wtp = self.Ht // scale, self.Wt // scale
        self.num_patches = self.Htp * self.Wtp
        self.subpatch_count = gs * gs * scale * scale

        self.patch_embed = layers.Conv2D(
            self.embed_dim,
            self.patch_size,
            strides=self.patch_size,
            use_bias=self.use_bias,
            name="patch_embed",
        )
        self.mlp_lin0 = layers.Dense(self.embed_dim * 2, name="mlp_lin0")
        self.mlp_ln0 = layers.LayerNormalization(epsilon=1e-5, name="mlp_ln0")
        self.mlp_lin1 = layers.Dense(self.embed_dim, name="mlp_lin1")
        self.mlp_ln1 = layers.LayerNormalization(epsilon=1e-5, name="mlp_ln1")
        super().build(input_shape)

    def call(self, x):
        conv_out = self.patch_embed(x)
        B = ops.shape(conv_out)[0]
        D, gs, scale = self.embed_dim, self.grid_size, self.scale

        t = _unfold(conv_out, 1, gs)
        t = _unfold(t, 2, gs)
        t = ops.reshape(t, (B, self.Ht, self.Wt, D, gs * gs))
        t = _unfold(t, 1, scale)
        t = _unfold(t, 2, scale)
        t = ops.reshape(t, (B, self.num_patches, D, gs * gs, scale, scale))
        t = ops.transpose(t, (0, 1, 2, 4, 5, 3))
        t = ops.reshape(t, (B, self.num_patches, D, self.subpatch_count))
        t = ops.transpose(t, (0, 1, 3, 2))
        t = ops.reshape(t, (B * self.num_patches, self.subpatch_count, D))

        t = ops.relu(self.mlp_ln0(self.mlp_lin0(t)))
        t = ops.relu(self.mlp_ln1(self.mlp_lin1(t)))
        return t


class AnySatTimeSeriesProjector(layers.Layer):

    def __init__(
        self,
        embed_dim,
        in_channels,
        T,
        in_norm,
        reduce_scale,
        scale,
        n_head=16,
        d_k=8,
        mlp_in=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.T_period = T
        self.use_in_norm = in_norm
        self.reduce_scale = reduce_scale
        self.scale = scale
        self.n_head = n_head
        self.d_k = d_k
        self.mlp_in_dims = (
            list(mlp_in)
            if mlp_in
            else [embed_dim // 8, embed_dim // 2, embed_dim, embed_dim * 2, embed_dim]
        )
        self.d_model = self.mlp_in_dims[-1]
        self.scale_eff = max(1, scale // reduce_scale)

    def build(self, input_shape):
        H, W = input_shape[2], input_shape[3]
        se = self.scale_eff
        self.H, self.W = H, W
        self.Hp, self.Wp = H // se, W // se
        self.num_patches = self.Hp * self.Wp

        dims = [self.in_channels] + self.mlp_in_dims
        self.inconv_lin = []
        self.inconv_gn = []
        for i in range(len(dims) - 1):
            self.inconv_lin.append(layers.Dense(dims[i + 1], name=f"inconv_lin{i}"))
            self.inconv_gn.append(
                layers.GroupNormalization(groups=4, epsilon=1e-5, name=f"inconv_gn{i}")
            )

        self.in_norm = (
            layers.GroupNormalization(groups=self.n_head, epsilon=1e-5, name="in_norm")
            if self.use_in_norm
            else None
        )

        self.fc1_k = layers.Dense(self.n_head * self.d_k, name="fc1_k")
        self.Q = self.add_weight(
            shape=(self.n_head, self.d_k), initializer="zeros", name="Q"
        )

        mlp_dims = [self.d_model, self.d_model]
        self.mlp_lin = [layers.Dense(mlp_dims[1], name="mlp_lin0")]
        self.mlp_gn = [
            layers.GroupNormalization(groups=4, epsilon=1e-5, name="mlp_gn0")
        ]

        self.out_norm = layers.GroupNormalization(
            groups=self.n_head, epsilon=1e-5, name="out_norm"
        )

        pe_dim = self.d_model // self.n_head
        omega = np.arange(pe_dim, dtype=np.float64)
        denom = self.T_period ** (2 * (omega // 2) / pe_dim)
        self._pe_denom = self.add_weight(
            shape=(pe_dim,),
            initializer=keras.initializers.Constant(denom),
            trainable=False,
            name="pe_denom",
        )
        self._pe_dim = pe_dim
        super().build(input_shape)

    def _positional_encoder(self, batch_positions):
        table = batch_positions[:, :, None] / self._pe_denom[None, None, :]
        sin_part = ops.sin(table)
        cos_part = ops.cos(table)
        idx = ops.arange(self._pe_dim)
        is_even = ops.equal(ops.mod(idx, 2), 0)
        out = ops.where(is_even[None, None, :], sin_part, cos_part)
        return ops.tile(out, (1, 1, self.n_head))

    def call(self, x, dates):
        B = ops.shape(x)[0]
        T = x.shape[1]
        H, W = self.H, self.W

        out = ops.transpose(x, (0, 2, 3, 1, 4))
        out = ops.reshape(out, (B * H * W, T, self.in_channels))

        d = ops.reshape(out, (-1, self.in_channels))
        for lin, gn in zip(self.inconv_lin, self.inconv_gn):
            d = ops.relu(gn(lin(d)))
        out = ops.reshape(d, (B * H * W, T, self.d_model))

        if self.in_norm is not None:
            out = self.in_norm(out)

        bp = ops.tile(dates[:, None, None, :], (1, H, W, 1))
        bp = ops.reshape(bp, (B * H * W, T))
        out = out + self._positional_encoder(bp)

        N = B * H * W
        n_head, d_k = self.n_head, self.d_k
        q = ops.tile(self.Q[:, None, :], (1, N, 1))
        q = ops.reshape(q, (n_head * N, d_k))

        k = self.fc1_k(out)
        k = ops.reshape(k, (N, T, n_head, d_k))
        k = ops.transpose(k, (2, 0, 1, 3))
        k = ops.reshape(k, (n_head * N, T, d_k))

        v = ops.reshape(out, (N, T, n_head, self.d_model // n_head))
        v = ops.transpose(v, (2, 0, 1, 3))
        v = ops.reshape(v, (n_head * N, T, self.d_model // n_head))

        attn = ops.matmul(q[:, None, :], ops.transpose(k, (0, 2, 1))) / ops.sqrt(
            ops.cast(d_k, "float32")
        )
        attn = ops.softmax(attn, axis=-1)
        att_out = ops.matmul(attn, v)
        att_out = ops.reshape(att_out, (n_head, N, self.d_model // n_head))
        out = ops.transpose(att_out, (1, 0, 2))
        out = ops.reshape(out, (N, self.d_model))

        for lin, gn in zip(self.mlp_lin, self.mlp_gn):
            out = ops.relu(gn(lin(out)))
        out = self.out_norm(out)

        E = self.d_model
        out = ops.reshape(out, (B, H, W, E))

        se = self.scale_eff
        t = _unfold(out, 1, se)
        t = _unfold(t, 2, se)
        t = ops.reshape(t, (B, self.num_patches, E, se * se))
        t = ops.transpose(t, (0, 1, 3, 2))
        t = ops.reshape(t, (B * self.num_patches, se * se, E))
        return t


def _mlp(x, fc1, fc2):
    return fc2(ops.gelu(fc1(x), approximate=False))


class AnySatPlainBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.mlp_ratio = mlp_ratio

    def build(self, input_shape):
        d = self.embed_dim
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.qkv = layers.Dense(d * 3, name="attn_qkv")
        self.proj = layers.Dense(d, name="attn_proj")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.fc1 = layers.Dense(int(d * self.mlp_ratio), name="mlp_fc1")
        self.fc2 = layers.Dense(d, name="mlp_fc2")
        super().build(input_shape)

    def call(self, x):
        B, N, C = ops.shape(x)[0], x.shape[1], self.embed_dim
        h = self.norm1(x)
        qkv = ops.reshape(self.qkv(h), (B, N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ops.softmax(ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, N, C))
        x = x + self.proj(out)
        x = x + _mlp(self.norm2(x), self.fc1, self.fc2)
        return x


class AnySatLocalEncoder(layers.Layer):

    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.mlp_ratio = mlp_ratio

    def build(self, input_shape):
        d = self.embed_dim
        self.cls_token = self.add_weight(
            shape=(1, 1, d), initializer="zeros", name="cls_token"
        )
        self.blocks = []
        for i in range(self.depth):
            blk = {
                "norm1": layers.LayerNormalization(
                    epsilon=1e-5, name=f"block{i}_norm1"
                ),
                "qkv": layers.Dense(d * 3, name=f"block{i}_attn_qkv"),
                "proj": layers.Dense(d, name=f"block{i}_attn_proj"),
                "norm2": layers.LayerNormalization(
                    epsilon=1e-5, name=f"block{i}_norm2"
                ),
                "fc1": layers.Dense(int(d * self.mlp_ratio), name=f"block{i}_mlp_fc1"),
                "fc2": layers.Dense(d, name=f"block{i}_mlp_fc2"),
            }
            rpe_w = self.add_weight(
                shape=(1, self.head_dim, _NUM_BUCKETS_NOSKIP + 1),
                initializer="zeros",
                name=f"block{i}_rpe_k_weight",
            )
            blk["rpe_k_weight"] = rpe_w
            self.blocks.append(blk)
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")
        super().build(input_shape)

    def _attn(self, x, blk, flat_idx, num_buckets, Lq, Lk):
        B, N, C = ops.shape(x)[0], x.shape[1], self.embed_dim
        qkv = ops.reshape(blk["qkv"](x), (B, N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        attn = attn + _rpe_bias_k(q, blk["rpe_k_weight"], flat_idx, num_buckets, Lq, Lk)
        attn = ops.softmax(attn)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, N, C))
        return blk["proj"](out)

    def call(self, tokens, pos_embed, flat_idx, num_buckets, Lq, Lk):
        B_ = ops.shape(tokens)[0]
        C = self.embed_dim
        cls = ops.broadcast_to(self.cls_token, (B_, 1, C))
        x = ops.concatenate([cls, tokens], axis=1)
        x = x + pos_embed[None]

        for blk in self.blocks:
            x = x + self._attn(blk["norm1"](x), blk, flat_idx, num_buckets, Lq, Lk)
            x = x + _mlp(blk["norm2"](x), blk["fc1"], blk["fc2"])
        x = self.norm(x)
        return x[:, 0]


class AnySatCrossPoolingBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.mlp_ratio = mlp_ratio

    def build(self, input_shape):
        d = self.embed_dim
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.wk = layers.Dense(d, name="attn_wk")
        self.wv = layers.Dense(d, name="attn_wv")
        self.q_learned = self.add_weight(
            shape=(1, 1, d), initializer="zeros", name="attn_q_learned"
        )
        self.rpe_k_weight = self.add_weight(
            shape=(1, self.head_dim, _NUM_BUCKETS_NOSKIP + 1),
            initializer="zeros",
            name="attn_rpe_k_weight",
        )
        self.proj = layers.Dense(d, name="attn_proj")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.fc1 = layers.Dense(int(d * self.mlp_ratio), name="mlp_fc1")
        self.fc2 = layers.Dense(d, name="mlp_fc2")
        super().build(input_shape)

    def call(
        self,
        x,
        pos_embed_q,
        n_modalities,
        num_patches_side,
        flat_idx,
        num_buckets,
        modis=False,
    ):
        h = self.norm1(x)
        B, N, C = ops.shape(h)[0], h.shape[1], self.embed_dim
        num_patches = num_patches_side * num_patches_side
        modis_i = int(modis)
        Nq = num_patches + 1 + modis_i

        q_ = self.q_learned + pos_embed_q[None]
        q_ = ops.broadcast_to(q_, (B, Nq, C))
        q = ops.transpose(
            ops.reshape(q_, (B, Nq, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        k = ops.transpose(
            ops.reshape(self.wk(h), (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        v = ops.transpose(
            ops.reshape(self.wv(h), (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        rpe = _rpe_bias_k(q, self.rpe_k_weight, flat_idx, num_buckets, Nq, Nq)
        rpe_prefix = rpe[:, :, :, : 1 + modis_i]
        rpe_tiled = ops.tile(rpe[:, :, :, 1 + modis_i :], (1, 1, 1, n_modalities))
        attn = attn + ops.concatenate([rpe_prefix, rpe_tiled], axis=-1)
        attn = ops.softmax(attn)

        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, Nq, C))
        out = ops.concatenate([out[:, :1], out[:, 1 + modis_i :]], axis=1)
        pooled = self.proj(out)

        x = pooled + _mlp(self.norm2(pooled), self.fc1, self.fc2)
        return x


class AnySatRelease(keras.Model):

    def __init__(
        self,
        modalities,
        input_shapes,
        scale,
        size="base",
        embed_dim=None,
        depth=None,
        num_heads=None,
        mlp_ratio=4.0,
        output="patch",
        name="anysat_release",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        cfg = ANYSAT_CONFIGS[size]
        self.embed_dim = embed_dim or cfg["embed_dim"]
        self.depth = depth or cfg["depth"]
        self.num_heads = num_heads or cfg["num_heads"]
        self.mlp_ratio = mlp_ratio
        self.modalities = list(modalities)
        self.input_shapes = dict(input_shapes)
        self.scale = scale
        self.output_mode = output
        if output not in ("tile", "patch"):
            raise ValueError(
                "output must be 'tile' or 'patch' (dense/subpatch output is not implemented)"
            )

        proj_cfgs = anysat_projector_configs(self.embed_dim)
        self.projectors = {}
        for m in self.modalities:
            pc = proj_cfgs[m]
            if pc["kind"] == "image":
                self.projectors[m] = AnySatImageProjector(
                    self.embed_dim,
                    pc["patch_size"],
                    pc["in_chans"],
                    pc["resolution"],
                    scale,
                    bias=False,
                    name=f"projector_{m}",
                )
            else:
                self.projectors[m] = AnySatTimeSeriesProjector(
                    self.embed_dim,
                    pc["in_channels"],
                    pc["T"],
                    pc["in_norm"],
                    pc["reduce_scale"],
                    scale,
                    n_head=_TS_LTAE_HEAD["n_head"],
                    d_k=_TS_LTAE_HEAD["d_k"],
                    mlp_in=pc["mlp_in"],
                    name=f"projector_{m}",
                )

        self.local_encoder = AnySatLocalEncoder(
            self.embed_dim,
            self.depth,
            self.num_heads,
            mlp_ratio,
            name="spatial_encoder",
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim), initializer="zeros", name="cls_token"
        )
        self.blocks = [
            AnySatPlainBlock(
                self.embed_dim, self.num_heads, mlp_ratio, name=f"block{i}"
            )
            for i in range(self.depth)
        ]
        self.cross_pool = AnySatCrossPoolingBlock(
            self.embed_dim, self.num_heads, mlp_ratio, name="block_cross"
        )

        self._build_meta()

    def _build_meta(self):
        proj_cfgs = anysat_projector_configs(self.embed_dim)
        num_patches_side = None
        self._local_pos_embed = {}
        self._local_rpe = {}
        for m in self.modalities:
            shape = self.input_shapes[m]
            pc = proj_cfgs[m]
            if pc["kind"] == "image":
                res = int(10 / pc["resolution"])
                gs = res // pc["patch_size"]
                Hc = shape[0] // pc["patch_size"]
                Ht = Hc // gs
                Htp = Ht // self.scale
                subpatch_count = gs * gs * self.scale * self.scale
            else:
                se = max(1, self.scale // pc["reduce_scale"])
                Htp = shape[1] // se
                subpatch_count = se * se
            side = int(round(subpatch_count**0.5))
            if side * side != subpatch_count:
                raise ValueError(
                    f"modality '{m}': subpatch count {subpatch_count} is not a perfect square"
                )
            pe, grid_aug = pos_embed_with_resolution(
                self.embed_dim, self.scale, ANYSAT_INPUT_RES[m]
            )
            if grid_aug != side:
                raise ValueError(
                    f"modality '{m}': local grid size mismatch ({grid_aug} from resolution vs {side} "
                    f"from actual subpatch count) - input_shapes/config are inconsistent for this modality"
                )
            self._local_pos_embed[m] = ops.convert_to_tensor(pe)
            flat_idx, num_buckets, Lq, Lk = _bucket_flat_offset_index(
                side, side, skip=1
            )
            self._local_rpe[m] = (ops.convert_to_tensor(flat_idx), num_buckets, Lq, Lk)

            if m == "modis":
                continue
            if num_patches_side is None:
                num_patches_side = Htp
            elif num_patches_side != Htp:
                raise ValueError(
                    f"modality '{m}' produces a {Htp}x{Htp} patch grid, inconsistent with "
                    f"{num_patches_side}x{num_patches_side} from an earlier modality - all modalities "
                    f"must cover the same footprint at the same `scale`"
                )
        self.num_patches_side = num_patches_side
        self._global_pos_embed = ops.convert_to_tensor(
            pos_embed_with_scale(
                self.embed_dim, num_patches_side, self.scale, cls_token=True
            )
        )
        self._modis = "modis" in self.modalities
        self._n_modalities = len([m for m in self.modalities if m != "modis"])
        flat_idx, num_buckets, Lq, Lk = _bucket_flat_offset_index(
            num_patches_side, num_patches_side, skip=1 + int(self._modis)
        )
        self._cross_rpe = (ops.convert_to_tensor(flat_idx), num_buckets, Lq, Lk)
        self._cross_pos_embed_q = ops.convert_to_tensor(
            pos_embed_with_scale(
                self.embed_dim,
                num_patches_side,
                self.scale,
                cls_token=True,
                modis=self._modis,
            )
        )

    def call(self, inputs):
        modality_order = [m for m in self.modalities if m in inputs]
        if not modality_order:
            raise ValueError("no known modality found in `inputs`")

        modis_tokens = None
        patch_tokens = {}
        batch_size = None
        for m in modality_order:
            raw = inputs[m]
            proj = self.projectors[m]
            tokens = proj(*raw) if isinstance(raw, (tuple, list)) else proj(raw)
            flat_idx, num_buckets, Lq, Lk = self._local_rpe[m]
            pooled = self.local_encoder(
                tokens,
                pos_embed=self._local_pos_embed[m],
                flat_idx=flat_idx,
                num_buckets=num_buckets,
                Lq=Lq,
                Lk=Lk,
            )
            if m == "modis":
                modis_tokens = pooled
                if batch_size is None:
                    batch_size = ops.shape(pooled)[0]
            else:
                num_patches = self.num_patches_side * self.num_patches_side
                pooled = ops.reshape(pooled, (-1, num_patches, self.embed_dim))
                batch_size = ops.shape(pooled)[0]
                patch_tokens[m] = pooled + self._global_pos_embed[None, 1:, :]

        cls = ops.broadcast_to(
            self.cls_token + self._global_pos_embed[None, :1, :],
            (batch_size, 1, self.embed_dim),
        )
        seq = [cls]
        if modis_tokens is not None:
            seq.append(ops.reshape(modis_tokens, (batch_size, 1, self.embed_dim)))
        for m in modality_order:
            if m != "modis":
                seq.append(patch_tokens[m])
        tokens = ops.concatenate(seq, axis=1)

        for blk in self.blocks:
            tokens = blk(tokens)

        flat_idx, num_buckets, Lq, Lk = self._cross_rpe
        pos_embed_q = self._cross_pos_embed_q
        tokens = self.cross_pool(
            tokens,
            pos_embed_q=pos_embed_q,
            n_modalities=self._n_modalities,
            num_patches_side=self.num_patches_side,
            flat_idx=flat_idx,
            num_buckets=num_buckets,
            modis=self._modis,
        )

        if self.output_mode == "tile":
            return tokens[:, 0, :]
        patch = tokens[:, 1:, :]
        return ops.reshape(
            patch,
            (batch_size, self.num_patches_side, self.num_patches_side, self.embed_dim),
        )
