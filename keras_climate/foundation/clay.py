"""
keras_climate.foundation.clay
----------------------------------
Clay (Clay Foundation, 2024): a ViT-MAE-style foundation model designed to
generalize across sensors by conditioning patch embeddings on per-band
metadata (wavelength) and per-sample geo-temporal metadata (lat/lon,
time-of-year), rather than assuming a fixed fixed band order/count like
most single-sensor foundation models.
"""

import math
import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import PatchEmbed2D, TransformerEncoderBlock


def fourier_encode(value, num_freqs=6):
    """Fourier feature encoding for a scalar metadata value (lat/lon/time),
    returns (..., 2*num_freqs)."""
    freqs = 2.0 ** np.arange(num_freqs).astype(np.float32) * math.pi
    args = value[..., None] * freqs
    return ops.concatenate([ops.sin(args), ops.cos(args)], axis=-1)


class BandEmbedding(layers.Layer):
    """Learns an embedding per spectral band from its (continuous)
    wavelength in micrometers, via an MLP over Fourier features - so new
    sensors/bands not seen in training can still be embedded sensibly."""

    def __init__(self, dim, num_freqs=6, **kwargs):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        self.mlp = keras.Sequential([
            layers.Dense(dim, activation="gelu"),
            layers.Dense(dim),
        ], name="band_mlp")

    def call(self, wavelengths):
        # wavelengths: (num_bands,) float
        feats = fourier_encode(wavelengths, self.num_freqs)
        return self.mlp(feats)  # (num_bands, dim)


class GeoTemporalEmbedding(layers.Layer):
    """Encodes (lat, lon, week_of_year) metadata into a single conditioning
    vector added to the class token."""

    def __init__(self, dim, num_freqs=6, **kwargs):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        self.mlp = keras.Sequential([
            layers.Dense(dim, activation="gelu"),
            layers.Dense(dim),
        ], name="geo_mlp")

    def call(self, latlon_time):
        # latlon_time: (B, 3) -> [lat_norm, lon_norm, week_norm], each pre-scaled to [-1, 1]
        feats = fourier_encode(latlon_time, self.num_freqs)  # (B, 3, 2*num_freqs)
        B = ops.shape(feats)[0]
        feats = ops.reshape(feats, (B, -1))
        return self.mlp(feats)


class ClayEncoder(keras.Model):
    """Metadata-conditioned ViT encoder. `call` takes a dict:
        {"pixels": (B, H, W, num_bands), "wavelengths": (num_bands,),
         "latlon_time": (B, 3)}
    Each band is patch-embedded independently (shared conv weights across
    bands via depthwise-style looping) then summed with its band embedding,
    so the token count / channel semantics generalize across sensors."""

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, name="clay_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid = img_size // patch_size

        # Single-channel patch projection, shared across bands (Clay embeds
        # each band through the same spatial projection then adds a
        # band-specific embedding, rather than learning per-sensor conv
        # kernels over a fixed band order).
        self.patch_proj = layers.Conv2D(embed_dim, patch_size, strides=patch_size, name="patch_proj")
        self.band_embed = BandEmbedding(embed_dim, name="band_embed")
        self.geo_embed = GeoTemporalEmbedding(embed_dim, name="geo_embed")
        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer="zeros",
                                          trainable=True, name="cls_token")
        self.spatial_pos = self.add_weight(shape=(self.grid * self.grid, embed_dim),
                                            initializer="zeros", trainable=True, name="spatial_pos")

        self.blocks = [TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, name=f"block{i}")
                        for i in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, inputs, training=False):
        pixels = inputs["pixels"]          # (B, H, W, num_bands)
        wavelengths = inputs["wavelengths"]  # (num_bands,)
        latlon_time = inputs["latlon_time"]  # (B, 3)

        num_bands = pixels.shape[-1]
        band_embeds = self.band_embed(wavelengths)  # (num_bands, D)

        band_tokens = []
        for b in range(num_bands):
            band_img = pixels[..., b:b + 1]  # (B, H, W, 1) - single-channel input
            tok = self.patch_proj(band_img)  # (B, grid, grid, D) via a 1-in-channel conv
            B = ops.shape(tok)[0]
            tok = ops.reshape(tok, (B, self.grid * self.grid, self.embed_dim))
            tok = tok + self.spatial_pos[None] + band_embeds[b][None, None, :]
            band_tokens.append(tok)

        tokens = ops.concatenate(band_tokens, axis=1)  # (B, num_bands*grid*grid, D)

        geo_cond = self.geo_embed(latlon_time)  # (B, D)
        B = ops.shape(tokens)[0]
        cls = ops.broadcast_to(self.cls_token, (B, 1, self.embed_dim)) + geo_cond[:, None, :]
        tokens = ops.concatenate([cls, tokens], axis=1)

        for blk in self.blocks:
            tokens = blk(tokens, training=training)
        tokens = self.norm(tokens)
        return tokens


def ClayClassifier(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                    num_bands=10, num_classes=10, name="clay_classifier"):
    pixels_in = keras.Input((img_size, img_size, num_bands), name="pixels")
    wavelengths_in = keras.Input((num_bands,), batch_size=1, name="wavelengths")
    latlon_time_in = keras.Input((3,), name="latlon_time")

    encoder = ClayEncoder(img_size, patch_size, embed_dim, depth, num_heads, name="encoder")
    tokens = encoder({
        "pixels": pixels_in,
        "wavelengths": layers.Lambda(lambda t: t[0])(wavelengths_in),
        "latlon_time": latlon_time_in,
    })
    cls_token = layers.Lambda(lambda t: t[:, 0], name="take_cls")(tokens)
    outputs = layers.Dense(num_classes, name="head")(cls_token)
    return keras.Model([pixels_in, wavelengths_in, latlon_time_in], outputs, name=name)
