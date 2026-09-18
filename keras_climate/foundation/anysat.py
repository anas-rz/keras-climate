"""
keras_climate.foundation.anysat
------------------------------------
AnySat (Astruc et al. 2024): a single ViT that ingests an arbitrary set of
modalities, each at its own native ground-sample distance (GSD) and patch
size, by projecting every modality's patches into a shared embedding space
and tagging each token with a learned modality embedding plus a
resolution-aware position embedding (patch center coordinates scaled by
GSD, so tokens from different resolutions/modalities are spatially
comparable). This lets one model jointly consume, e.g., 10m Sentinel-2,
20m Sentinel-1, and 30m Landsat over the same footprint.
"""

import keras
from keras import layers, ops
from keras_climate.utils.layers import TransformerEncoderBlock, MLP


class ModalityPatchEmbed(layers.Layer):
    """Patch-embeds one modality's imagery at its native patch size, and
    tags each token with a continuous (x, y) center coordinate in meters
    (patch_size_px * gsd_m_per_px), so tokens are comparable across
    modalities regardless of native resolution."""

    def __init__(self, embed_dim, patch_size_px, gsd_m, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size_px = patch_size_px
        self.gsd_m = gsd_m

    def build(self, input_shape):
        self.proj = layers.Conv2D(self.embed_dim, self.patch_size_px,
                                   strides=self.patch_size_px, name="proj")
        super().build(input_shape)

    def call(self, x):
        tok = self.proj(x)
        H, W = ops.shape(tok)[1], ops.shape(tok)[2]
        B = ops.shape(tok)[0]
        tok = ops.reshape(tok, (B, H * W, self.embed_dim))

        patch_extent_m = self.patch_size_px * self.gsd_m
        ys = (ops.arange(H, dtype="float32") + 0.5) * patch_extent_m
        xs = (ops.arange(W, dtype="float32") + 0.5) * patch_extent_m
        grid_y, grid_x = ops.meshgrid(ys, xs, indexing="ij")
        coords = ops.stack([ops.reshape(grid_x, (-1,)), ops.reshape(grid_y, (-1,))], axis=-1)  # (H*W, 2)
        coords = ops.broadcast_to(coords[None], (B, H * W, 2))

        return tok, coords


class ContinuousPositionEncoding(layers.Layer):
    """Maps a continuous (x, y) coordinate in meters to a `dim`-length
    embedding via an MLP over sinusoidal features - resolution/modality
    agnostic, unlike a learned grid position embedding."""

    def __init__(self, dim, num_freqs=8, max_freq=1024.0, **kwargs):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        self.max_freq = max_freq
        self.mlp = MLP(dim, dim, name="coord_mlp")

    def call(self, coords):
        # coords: (B, N, 2) in meters
        freqs = self.max_freq ** (ops.arange(self.num_freqs, dtype="float32") / self.num_freqs)
        args = coords[..., None] / freqs  # (B, N, 2, num_freqs)
        feats = ops.concatenate([ops.sin(args), ops.cos(args)], axis=-1)  # (B, N, 2, 2*num_freqs)
        B, N = ops.shape(coords)[0], ops.shape(coords)[1]
        feats = ops.reshape(feats, (B, N, 4 * self.num_freqs))
        return self.mlp(feats)


class AnySatEncoder(keras.Model):
    """Modality-agnostic ViT encoder.

    `modality_specs`: dict[name -> dict(patch_size_px=int, gsd_m=float)].
    `call(inputs)` takes a dict[name -> (B, H, W, C) tensor] with only the
    modalities present for this sample/batch (missing modalities are simply
    omitted from the dict - no fixed channel layout required)."""

    def __init__(self, modality_specs, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, name="anysat_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.modality_specs = modality_specs
        self.embed_dim = embed_dim

        self.patch_embeds = {
            m: ModalityPatchEmbed(embed_dim, spec["patch_size_px"], spec["gsd_m"], name=f"embed_{m}")
            for m, spec in modality_specs.items()
        }
        self.modality_tokens = {
            m: self.add_weight(shape=(1, 1, embed_dim), initializer="zeros", name=f"modtok_{m}")
            for m in modality_specs
        }
        self.pos_encoding = ContinuousPositionEncoding(embed_dim, name="pos_encoding")
        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer="zeros", name="cls_token")

        self.blocks = [TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, name=f"block{i}")
                        for i in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, inputs, training=False):
        all_tokens = []
        batch_size = None
        for modality, tensor in inputs.items():
            tok, coords = self.patch_embeds[modality](tensor)
            batch_size = ops.shape(tok)[0]
            tok = tok + self.pos_encoding(coords) + self.modality_tokens[modality]
            all_tokens.append(tok)

        tokens = ops.concatenate(all_tokens, axis=1)
        cls = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        tokens = ops.concatenate([cls, tokens], axis=1)

        for blk in self.blocks:
            tokens = blk(tokens, training=training)
        return self.norm(tokens)


def AnySatClassifier(modality_specs, embed_dim=768, depth=12, num_heads=12,
                      input_shapes=None, num_classes=10, name="anysat_classifier"):
    """`input_shapes`: dict[name -> (H, W, C)] declaring the Keras.Input
    shape for each modality this particular model instance will accept."""
    inputs = {m: keras.Input(shape=shape, name=m) for m, shape in input_shapes.items()}
    encoder = AnySatEncoder(modality_specs, embed_dim, depth, num_heads, name="encoder")
    tokens = encoder(inputs)
    cls_token = layers.Lambda(lambda t: t[:, 0], name="take_cls")(tokens)
    outputs = layers.Dense(num_classes, name="head")(cls_token)
    return keras.Model(inputs, outputs, name=name)
