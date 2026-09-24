import numpy as np
import keras
from keras import layers, ops
from keras_climate.utils.layers import (
    PatchEmbed2D,
    TransformerEncoderBlock,
    sincos_position_embedding_2d,
)


class MaskingLayer(layers.Layer):

    def __init__(self, mask_ratio=0.75, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.seed_generator = keras.random.SeedGenerator()

    def call(self, x):
        # N (token count) is fixed by the architecture (img_size/patch_size),
        # never batch-dependent, so keep it a static Python int rather than a
        # traced `ops.shape(x)[1]` value: `len_keep` is used below as a slice
        # bound, and JAX requires slice bounds to be static under tracing.
        B, N = ops.shape(x)[0], x.shape[1]
        len_keep = int(N * (1 - self.mask_ratio))

        noise = keras.random.uniform((B, N), seed=self.seed_generator)
        ids_shuffle = ops.argsort(noise, axis=1)
        ids_restore = ops.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = ops.take_along_axis(x, ids_keep[:, :, None], axis=1)

        mask = ops.ones((B, N))
        mask = ops.scatter_update(
            ops.reshape(mask, (-1,)),
            ops.reshape(ops.arange(B)[:, None] * N + ids_keep, (-1, 1)),
            ops.zeros((B * len_keep,)),
        )
        mask = ops.reshape(mask, (B, N))
        mask = ops.take_along_axis(mask, ids_restore, axis=1)

        return x_kept, mask, ids_restore


class SatMAEEncoder(keras.Model):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        mode="single",
        num_groups=3,
        num_frames=3,
        name="satmae_encoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mode = mode
        self.num_groups = num_groups
        self.num_frames = num_frames

        self.patch_embed = PatchEmbed2D(
            patch_size, embed_dim, norm=False, name="patch_embed"
        )
        self.cls_token = self.add_weight(
            shape=(1, 1, embed_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token",
        )

        num_patches = (img_size // patch_size) ** 2
        pos = sincos_position_embedding_2d(
            img_size // patch_size, img_size // patch_size, embed_dim
        )
        pos = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos], axis=0)
        self.pos_embed = self.add_weight(
            shape=pos.shape,
            initializer=keras.initializers.Constant(pos),
            trainable=False,
            name="pos_embed",
        )

        if mode == "multispectral":
            self.group_embed = self.add_weight(
                shape=(num_groups, 1, embed_dim),
                initializer="zeros",
                trainable=True,
                name="group_embed",
            )
        elif mode == "temporal":
            self.temporal_embed = self.add_weight(
                shape=(num_frames, 1, embed_dim),
                initializer="zeros",
                trainable=True,
                name="temporal_embed",
            )

        self.blocks = [
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, name=f"block{i}")
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def _embed_single(self, img):
        tokens, H, W = self.patch_embed(img)
        tokens = tokens + self.pos_embed[None, 1:, :]
        return tokens

    def call(self, x, training=False, apply_masking=False, mask_ratio=0.75):
        if self.mode == "single":
            tokens = self._embed_single(x)
        elif self.mode == "multispectral":
            group_tokens = []
            for g in range(self.num_groups):
                t = self._embed_single(x[:, g])
                t = t + self.group_embed[g][None]
                group_tokens.append(t)
            tokens = ops.concatenate(group_tokens, axis=1)
        elif self.mode == "temporal":
            frame_tokens = []
            for t_idx in range(self.num_frames):
                t = self._embed_single(x[:, t_idx])
                t = t + self.temporal_embed[t_idx][None]
                frame_tokens.append(t)
            tokens = ops.concatenate(frame_tokens, axis=1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        mask, ids_restore = None, None
        if apply_masking:
            tokens, mask, ids_restore = MaskingLayer(mask_ratio)(tokens)

        B = ops.shape(tokens)[0]
        cls = ops.broadcast_to(
            self.cls_token + self.pos_embed[None, :1, :], (B, 1, self.embed_dim)
        )
        tokens = ops.concatenate([cls, tokens], axis=1)

        for blk in self.blocks:
            tokens = blk(tokens, training=training)
        tokens = self.norm(tokens)

        if apply_masking:
            return tokens, mask, ids_restore
        return tokens


class SatMAEDecoder(keras.Model):

    def __init__(
        self,
        num_patches,
        patch_size,
        in_chans,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        encoder_embed_dim=768,
        num_repeats=1,
        name="satmae_decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.decoder_embed = layers.Dense(decoder_embed_dim, name="decoder_embed")
        self.mask_token = self.add_weight(
            shape=(1, 1, decoder_embed_dim),
            initializer="zeros",
            trainable=True,
            name="mask_token",
        )
        base_patches = num_patches // num_repeats
        grid_size = int(round(base_patches**0.5))
        if grid_size * grid_size != base_patches:
            raise ValueError(
                f"num_patches // num_repeats ({base_patches}) is not a perfect square; "
                "cannot build a 2D sin-cos grid position embedding."
            )
        base_pos = sincos_position_embedding_2d(grid_size, grid_size, decoder_embed_dim)
        pos = np.tile(base_pos, (num_repeats, 1))
        pos = np.concatenate(
            [np.zeros((1, decoder_embed_dim), dtype=np.float32), pos], axis=0
        )
        self.decoder_pos_embed = self.add_weight(
            shape=pos.shape,
            initializer=keras.initializers.Constant(pos),
            trainable=False,
            name="decoder_pos_embed",
        )
        self.blocks = [
            TransformerEncoderBlock(
                decoder_embed_dim, decoder_num_heads, name=f"block{i}"
            )
            for i in range(decoder_depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.pred = layers.Dense(patch_size * patch_size * in_chans, name="pred")
        self.num_patches = num_patches

    def call(self, x, ids_restore, training=False):
        x = self.decoder_embed(x)
        B = ops.shape(x)[0]
        num_mask = self.num_patches + 1 - ops.shape(x)[1]
        mask_tokens = ops.broadcast_to(self.mask_token, (B, num_mask, x.shape[-1]))
        x_ = ops.concatenate([x[:, 1:, :], mask_tokens], axis=1)
        x_ = ops.take_along_axis(x_, ids_restore[:, :, None], axis=1)
        x = ops.concatenate([x[:, :1, :], x_], axis=1)
        x = x + self.decoder_pos_embed[None]

        for blk in self.blocks:
            x = blk(x, training=training)
        x = self.norm(x)
        x = self.pred(x)
        return x[:, 1:, :]


def SatMAE(
    img_size=224,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mode="single",
    num_groups=3,
    num_frames=3,
    mask_ratio=0.75,
    name="satmae",
):
    if mode == "single":
        inp_shape = (img_size, img_size, in_chans)
    elif mode == "multispectral":
        inp_shape = (num_groups, img_size, img_size, in_chans)
    else:
        inp_shape = (num_frames, img_size, img_size, in_chans)

    inputs = keras.Input(shape=inp_shape, name="image")
    encoder = SatMAEEncoder(
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        depth,
        num_heads,
        mode=mode,
        num_groups=num_groups,
        num_frames=num_frames,
        name=f"{name}_encoder",
    )
    tokens, mask, ids_restore = encoder(
        inputs, apply_masking=True, mask_ratio=mask_ratio
    )

    num_patches = (img_size // patch_size) ** 2
    num_repeats = 1
    if mode == "multispectral":
        num_repeats = num_groups
    elif mode == "temporal":
        num_repeats = num_frames
    num_patches *= num_repeats

    decoder = SatMAEDecoder(
        num_patches,
        patch_size,
        in_chans,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        embed_dim,
        num_repeats=num_repeats,
        name=f"{name}_decoder",
    )
    pred = decoder(tokens, ids_restore)

    return keras.Model(inputs, [pred, mask], name=name)
