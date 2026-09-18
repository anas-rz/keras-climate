"""
keras_climate.remote_sensing.segformer
-----------------------------------------
SegFormer (Xie et al. 2021): hierarchical Mix Transformer (MiT) encoder with
efficient spatial-reduction attention + Mix-FFN, and a lightweight all-MLP
decode head. Strong general-purpose choice for satellite/aerial segmentation.
"""

import keras
from keras import layers, ops
from ..utils.layers import OverlapPatchEmbed, SegformerBlock, ConvBNAct


MIT_CONFIGS = {
    # embed_dims, depths, num_heads, sr_ratios, mlp_ratio
    "b0": dict(embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2],
               num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], decoder_dim=256),
    "b1": dict(embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2],
               num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], decoder_dim=256),
    "b2": dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3],
               num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], decoder_dim=768),
    "b3": dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3],
               num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], decoder_dim=768),
    "b4": dict(embed_dims=[64, 128, 320, 512], depths=[3, 8, 27, 3],
               num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], decoder_dim=768),
    "b5": dict(embed_dims=[64, 128, 320, 512], depths=[3, 6, 40, 3],
               num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], decoder_dim=768),
}


class MiTStage(layers.Layer):
    """One MiT encoder stage: overlap patch embed -> N SegFormer blocks ->
    norm -> reshape back to a spatial (B, H, W, C) map.

    Keeping the whole stage (including the intermediate token sequence and
    its H/W bookkeeping) inside a single layer's `call()` matters: those
    intermediates never cross the Functional-API tracing boundary, unlike a
    plain function that would `return tokens, H, W` from a layer call - the
    Functional API requires layer outputs to be tensors, and static H/W
    ints resolved at graph-build time are not tensors.
    """

    def __init__(self, patch_size, stride, embed_dim, num_heads, depth, sr_ratio,
                 mlp_ratio=4.0, drop_path_rates=None, name="mit_stage", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = embed_dim
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.patch_embed = OverlapPatchEmbed(patch_size, stride, embed_dim, name="patch_embed")
        self.blocks = [
            SegformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, sr_ratio=sr_ratio,
                            drop_path=drop_path_rates[d], name=f"block{d}")
            for d in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, x, training=False):
        tokens, H, W = self.patch_embed(x)
        for blk in self.blocks:
            tokens = blk(tokens, H=H, W=W, training=training)
        tokens = self.norm(tokens)
        B, C = ops.shape(tokens)[0], self.embed_dim
        return ops.reshape(tokens, (B, H, W, C))


def mit_encoder(x, cfg, drop_path_rate=0.1, name="mit"):
    """Returns a list of 4 multi-scale feature maps (NHWC), one per stage."""
    patch_sizes = [7, 3, 3, 3]
    strides = [4, 2, 2, 2]
    total_blocks = sum(cfg["depths"])
    dpr = [drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]
    block_idx = 0

    features = []
    for stage in range(4):
        depth = cfg["depths"][stage]
        stage_dpr = dpr[block_idx:block_idx + depth]
        block_idx += depth

        x = MiTStage(
            patch_sizes[stage], strides[stage], cfg["embed_dims"][stage],
            cfg["num_heads"][stage], depth, cfg["sr_ratios"][stage],
            mlp_ratio=4.0, drop_path_rates=stage_dpr,
            name=f"{name}_stage{stage+1}",
        )(x)
        features.append(x)

    return features


def segformer_decode_head(features, decoder_dim, num_classes, target_hw, name="decode_head"):
    """All-MLP decoder: project each scale to a common dim, upsample to
    1/4 resolution, concat, fuse, classify."""
    h4, w4 = ops.shape(features[0])[1], ops.shape(features[0])[2]
    projected = []
    for i, f in enumerate(features):
        p = layers.Dense(decoder_dim, name=f"{name}_linear_c{i+1}")(f)
        p = layers.Lambda(
            lambda t, h=h4, w=w4: ops.image.resize(t, (h, w), interpolation="bilinear"),
            name=f"{name}_upsample{i+1}",
        )(p)
        projected.append(p)

    x = layers.Concatenate(axis=-1, name=f"{name}_concat")(list(reversed(projected)))
    x = ConvBNAct(decoder_dim, 1, name=f"{name}_fuse")(x)
    x = layers.Dropout(0.1, name=f"{name}_drop")(x)
    x = layers.Conv2D(num_classes, 1, name=f"{name}_classifier")(x)
    x = layers.Lambda(
        lambda t: ops.image.resize(t, target_hw, interpolation="bilinear"),
        name=f"{name}_upsample_to_input",
    )(x)
    return x


def SegFormer(
    input_shape=(512, 512, 3),
    num_classes=1,
    variant="b0",
    final_activation=None,
    name="segformer",
):
    cfg = MIT_CONFIGS[variant]
    inputs = keras.Input(shape=input_shape, name="image")

    features = mit_encoder(inputs, cfg, name=f"{name}_backbone")
    x = segformer_decode_head(features, cfg["decoder_dim"], num_classes,
                               target_hw=(input_shape[0], input_shape[1]),
                               name=f"{name}_decode_head")

    if final_activation:
        x = layers.Activation(final_activation, name="logits_act")(x)

    return keras.Model(inputs, x, name=f"{name}_{variant}")
