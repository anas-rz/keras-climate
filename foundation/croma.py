"""
keras_climate.foundation.croma
-----------------------------------
CROMA (Fuller et al. 2023): a dual-encoder foundation model that jointly
represents SAR (e.g. Sentinel-1) and optical (e.g. Sentinel-2) imagery.
Each modality has its own ViT encoder; a cross-attention fusion stack
produces a joint multimodal representation. Pretraining combines a
contrastive objective (aligning the two modalities' global representations)
with per-modality masked-image modeling; this module exposes the encoders
and fusion block so either objective can be wired up externally, plus a
convenience `CROMA(...)` model that returns unimodal + joint embeddings.
"""

import keras
from keras import layers, ops
from ..utils.layers import PatchEmbed2D, TransformerEncoderBlock


class ModalityEncoder(keras.Model):
    """A standard ViT encoder for one modality (SAR or optical)."""

    def __init__(self, img_size=120, patch_size=8, in_chans=2, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, name="modality_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        self.grid = img_size // patch_size
        self.patch_embed = PatchEmbed2D(patch_size, embed_dim, norm=False, name="patch_embed")
        self.pos_embed = self.add_weight(shape=(self.grid * self.grid, embed_dim),
                                          initializer="zeros", trainable=True, name="pos_embed")
        self.blocks = [TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, name=f"block{i}")
                        for i in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, x, training=False):
        tokens, H, W = self.patch_embed(x)
        tokens = tokens + self.pos_embed[None]
        for blk in self.blocks:
            tokens = blk(tokens, training=training)
        return self.norm(tokens)  # (B, N, D) - no cls token; CROMA pools via attention


class CrossAttentionFusion(layers.Layer):
    """Bidirectional cross-attention: SAR tokens attend to optical tokens
    and vice versa, then both streams are concatenated and self-attended
    to produce the joint representation."""

    def __init__(self, dim, num_heads, depth=2, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.cross_sar_to_opt = [_CrossAttnBlock(dim, num_heads, mlp_ratio, name=f"s2o_{i}")
                                  for i in range(depth)]
        self.cross_opt_to_sar = [_CrossAttnBlock(dim, num_heads, mlp_ratio, name=f"o2s_{i}")
                                  for i in range(depth)]
        self.joint_blocks = [TransformerEncoderBlock(dim, num_heads, mlp_ratio, name=f"joint_{i}")
                              for i in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="joint_norm")

    def call(self, sar_tokens, opt_tokens, training=False):
        s, o = sar_tokens, opt_tokens
        for blk in self.cross_sar_to_opt:
            s = blk(s, o, training=training)
        for blk in self.cross_opt_to_sar:
            o = blk(o, s, training=training)

        joint = ops.concatenate([s, o], axis=1)
        for blk in self.joint_blocks:
            joint = blk(joint, training=training)
        return self.norm(joint)


class _CrossAttnBlock(layers.Layer):
    """Pre-norm cross-attention block: query stream attends to a separate
    key/value (context) stream."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm_q = layers.LayerNormalization(epsilon=1e-6, name="norm_q")
        self.norm_kv = layers.LayerNormalization(epsilon=1e-6, name="norm_kv")
        self.q = layers.Dense(dim, name="q")
        self.kv = layers.Dense(dim * 2, name="kv")
        self.proj = layers.Dense(dim, name="proj")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        from ..utils.layers import MLP
        self.mlp = MLP(int(dim * mlp_ratio), dim, name="mlp")

    def call(self, x, context, training=False):
        B, N = ops.shape(x)[0], ops.shape(x)[1]
        Nc = ops.shape(context)[1]

        q = self.q(self.norm_q(x))
        q = ops.transpose(ops.reshape(q, (B, N, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        kv = self.kv(self.norm_kv(context))
        kv = ops.transpose(ops.reshape(kv, (B, Nc, 2, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = ops.softmax(ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (B, N, self.dim))
        out = self.proj(out)

        x = x + out
        x = x + self.mlp(self.norm2(x), training=training)
        return x


def CROMA(
    img_size=120,
    patch_size=8,
    sar_chans=2,
    optical_chans=12,
    embed_dim=768,
    encoder_depth=12,
    fusion_depth=2,
    num_heads=12,
    name="croma",
):
    """Returns a model mapping {sar, optical} -> {sar_repr, optical_repr,
    joint_repr}, where sar_repr/optical_repr are mean-pooled unimodal
    embeddings (for the contrastive objective) and joint_repr is the fused
    multimodal token sequence (for downstream dense/fusion tasks)."""
    sar_in = keras.Input((img_size, img_size, sar_chans), name="sar")
    opt_in = keras.Input((img_size, img_size, optical_chans), name="optical")

    sar_encoder = ModalityEncoder(img_size, patch_size, sar_chans, embed_dim,
                                   encoder_depth, num_heads, name="sar_encoder")
    opt_encoder = ModalityEncoder(img_size, patch_size, optical_chans, embed_dim,
                                   encoder_depth, num_heads, name="optical_encoder")

    sar_tokens = sar_encoder(sar_in)
    opt_tokens = opt_encoder(opt_in)

    fusion = CrossAttentionFusion(embed_dim, num_heads, fusion_depth, name="fusion")
    joint_tokens = fusion(sar_tokens, opt_tokens)

    sar_repr = layers.GlobalAveragePooling1D(name="sar_pool")(sar_tokens)
    opt_repr = layers.GlobalAveragePooling1D(name="optical_pool")(opt_tokens)

    return keras.Model(
        [sar_in, opt_in],
        {"sar_repr": sar_repr, "optical_repr": opt_repr, "joint_tokens": joint_tokens},
        name=name,
    )
