import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List

from keras_climate.weights.converter import WeightConverter, load_torch_state_dict_as_numpy, load_safetensors_as_numpy
from keras_climate.weights.mappings import (
    build_vit_mapper, build_unet_mapper, build_deeplabv3plus_mapper,
    build_segformer_mapper, build_segformer_param_kind_map, build_satmae_mapper,
)

from keras_climate.remote_sensing import UNet, DeepLabV3Plus, SegFormer, SatMAE, MIT_CONFIGS
from keras_climate.forecasting import PatchTST
from keras_climate.foundation import PrithviClassifier


@dataclass
class ModelSpec:
    build_fn: Callable[[], "keras.Model"]
    name_map_fn: Callable[[], Callable]
    loader: str = "torch"
    key_prefix_strip: Optional[str] = None
    param_kind_map: Dict[str, str] = field(default_factory=dict)
    skip_patterns: List[str] = field(default_factory=lambda: [
        r"^optimizer", r"\bnum_batches_tracked$", r"^ema_",
    ])


MODEL_REGISTRY = {
    "unet": ModelSpec(
        build_fn=lambda: UNet(input_shape=(256, 256, 3), num_classes=1),
        name_map_fn=build_unet_mapper,
        loader="torch",
    ),
    "deeplabv3plus_resnet50": ModelSpec(
        build_fn=lambda: DeepLabV3Plus(input_shape=(512, 512, 3), num_classes=19,
                                        backbone_layers=(3, 4, 6, 3), output_stride=16),
        name_map_fn=lambda: build_deeplabv3plus_mapper(layer_counts=(3, 4, 6, 3)),
        loader="torch",
    ),
    "deeplabv3plus_resnet101": ModelSpec(
        build_fn=lambda: DeepLabV3Plus(input_shape=(512, 512, 3), num_classes=19,
                                        backbone_layers=(3, 4, 23, 3), output_stride=16),
        name_map_fn=lambda: build_deeplabv3plus_mapper(layer_counts=(3, 4, 23, 3)),
        loader="torch",
    ),
    "segformer_b0": ModelSpec(
        build_fn=lambda: SegFormer(input_shape=(512, 512, 3), num_classes=19, variant="b0"),
        name_map_fn=lambda: build_segformer_mapper(MIT_CONFIGS["b0"]),
        loader="torch",
        param_kind_map=build_segformer_param_kind_map(MIT_CONFIGS["b0"]),
    ),
    "satmae_vitlarge": ModelSpec(
        build_fn=lambda: SatMAE(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16),
        name_map_fn=lambda: build_satmae_mapper(),
        loader="torch",
        key_prefix_strip="model.",
    ),
    "patchtst": ModelSpec(
        build_fn=lambda: PatchTST(seq_len=336, pred_len=96, num_channels=7),
        name_map_fn=lambda: build_vit_mapper("patchtst"),
        loader="torch",
    ),
    "prithvi_100m": ModelSpec(
        build_fn=lambda: PrithviClassifier(variant="prithvi_100m", num_classes=1000),
        name_map_fn=lambda: build_vit_mapper("encoder"),
        loader="torch",
        key_prefix_strip="encoder.",
        param_kind_map={
            "encoder/patch_embed/proj/kernel": "conv3d_kernel",
        },
    ),
}


def port(model_key, checkpoint_path, output_path, strict=False):
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_key}'. Available: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_key]

    model = spec.build_fn()
    if not model.built:
        raise RuntimeError(
            "Model is not built. For subclassed models, call it once on a "
            "dummy batch (matching the checkpoint's expected input shape) "
            "before porting weights."
        )

    if spec.loader == "torch":
        state_dict = load_torch_state_dict_as_numpy(checkpoint_path, spec.key_prefix_strip)
    elif spec.loader == "safetensors":
        state_dict = load_safetensors_as_numpy(checkpoint_path)
    else:
        raise ValueError(f"Unknown loader '{spec.loader}'")

    name_map = spec.name_map_fn()
    converter = WeightConverter(
        model, state_dict, name_map,
        param_kind_map=spec.param_kind_map,
        skip_patterns=spec.skip_patterns,
    )
    report = converter.convert(strict=strict, verbose=True)

    model.save_weights(output_path)
    print(f"[keras_climate] saved converted weights to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Port a pretrained checkpoint into keras_climate.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--checkpoint", required=True, help="Path to the source .pt/.pth/.bin/.safetensors file")
    parser.add_argument("--output", required=True, help="Output path, e.g. model.weights.h5")
    parser.add_argument("--strict", action="store_true",
                         help="Fail if any weight cannot be matched on either side")
    args = parser.parse_args()
    port(args.model, args.checkpoint, args.output, strict=args.strict)


if __name__ == "__main__":
    main()
