"""
keras_climate.weights.pretrained
-----------------------------------
"timm-style" pretrained-weight loaders for the `remote_sensing` models:
each function here downloads a real, publicly-hosted PyTorch checkpoint
(cached locally so it's only fetched once), ports it through this repo's
`WeightConverter` + the matching name-mapping, and returns a ready
`keras.Model` (or encoder/decoder pair) with real pretrained weights
loaded - no training required to get a working model.

    from keras_climate.weights.pretrained import unet_carvana
    model, report = unet_carvana()
    model.summary()

Every loader returns `(model, report)` (or `(encoder, decoder, report)` for
SatMAE) where `report` is the `WeightConverter.convert(...)` report dict,
so callers can inspect exactly what did/didn't get matched.

Coverage note - not every architecture in this repo has a publicly
downloadable checkpoint that matches it exactly:
  - `unet_carvana`: exact architecture + checkpoint match (milesial's own
    released Carvana weights).
  - `deeplabv3plus_resnet50_imagenet_backbone`: only the ResNet-50 backbone
    is pretrained (ImageNet-1k, via torchvision); no public checkpoint
    exists for this exact ASPP+decoder head, so it stays randomly
    initialized. Useful for verifying backbone-porting fidelity and as a
    fine-tuning starting point, not for out-of-the-box segmentation.
  - `segformer_b0_ade20k`: exact architecture + checkpoint match (the full
    encoder + decode head, fine-tuned on ADE20K) - requires `transformers`
    at conversion time to fetch/read the source checkpoint.
  - `satmae_vit_base_mae`: SatMAE's own checkpoints aren't hosted for
    direct download; `mode="single"` is architecturally identical to a
    plain ViT-MAE, so Meta AI's official ImageNet-1k MAE checkpoint is
    used as the best publicly-available stand-in for validating that
    shared core end-to-end.
  - `prithvi_eo_100m`: exact architecture + checkpoint match (IBM/NASA's
    own `Prithvi-EO-1.0-100M` release, encoder only - the MAE decoder
    portion of the checkpoint is skipped since this repo doesn't implement
    a Prithvi decoder).
  - `croma_base`/`croma_large`: exact architecture + checkpoint match (the
    paper authors' own released checkpoint, both modalities + the joint
    fusion encoder).

Clay and AnySat have no loader here: Clay's official checkpoint is ~5GB
(impractical to fetch/validate in most environments), and AnySat's
official checkpoint uses a substantially more complex, config-driven
architecture than `AnySatEncoder` implements (see
`keras_climate.foundation.anysat`'s module docstring) - both are still
validated against synthetic references matching their respective
mappings' assumed naming (see `foundation/test_clay.py` /
`foundation/test_anysat.py`), just not against the real public checkpoints.
"""

import os
import urllib.request

import numpy as np

from keras_climate.remote_sensing import UNet, DeepLabV3Plus, SegFormer, MIT_CONFIGS, SatMAEEncoder, SatMAEDecoder
from keras_climate.foundation import PrithviEncoder
from keras_climate.foundation.croma import CROMA
from keras_climate.weights.converter import WeightConverter, load_torch_state_dict_as_numpy
from keras_climate.weights.mappings import (
    build_unet_mapper,
    build_resnet_backbone_mapper,
    build_segformer_mapper,
    build_segformer_param_kind_map,
    convert_hf_segformer_state_dict,
    build_satmae_mapper,
    build_vit_mapper,
    load_croma_checkpoint,
    convert_croma_state_dict,
    build_croma_identity_mapper,
)

DEFAULT_CACHE_DIR = os.environ.get(
    "KERAS_CLIMATE_CACHE_DIR", os.path.expanduser("~/.cache/keras_climate/weights")
)


def _download(url, filename, cache_dir=None):
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def unet_carvana(input_shape=(256, 256, 3), cache_dir=None, strict=True):
    """UNet (base_filters=64, depth=4), binary car-vs-background
    segmentation (2 output classes/logits), pretrained on the Carvana
    dataset - the exact checkpoint from milesial/Pytorch-UNet's v3.0
    release."""
    path = _download(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/"
        "unet_carvana_scale1.0_epoch2.pth",
        "unet_carvana_scale1.0_epoch2.pth", cache_dir,
    )
    model = UNet(input_shape=input_shape, num_classes=2, base_filters=64, depth=4)
    model(np.zeros((1,) + input_shape, dtype="float32"))  # build

    state_dict = load_torch_state_dict_as_numpy(path)
    report = WeightConverter(
        model, state_dict, build_unet_mapper(depth=4),
        skip_patterns=[r"num_batches_tracked$"],
    ).convert(strict=strict, verbose=True)
    return model, report


def deeplabv3plus_resnet50_imagenet_backbone(input_shape=(512, 512, 3), num_classes=21, cache_dir=None):
    """DeepLabV3+ (ResNet-50, output_stride=16) with an ImageNet-1k
    pretrained torchvision ResNet-50 backbone; the ASPP + decoder head
    have no matching public checkpoint and stay randomly initialized (see
    module docstring)."""
    path = _download(
        "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "resnet50-11ad3fa6.pth", cache_dir,
    )
    model = DeepLabV3Plus(input_shape=input_shape, num_classes=num_classes,
                           backbone_layers=(3, 4, 6, 3), output_stride=16)
    model(np.zeros((1,) + input_shape, dtype="float32"))  # build

    state_dict = load_torch_state_dict_as_numpy(path)
    mapper = build_resnet_backbone_mapper(layer_counts=(3, 4, 6, 3))
    # strict=False: the checkpoint only covers the backbone (no ASPP/decoder
    # weights exist to match); `fc.*` (the ImageNet classifier head) and
    # `num_batches_tracked` (a BN counter with no Keras equivalent) are
    # expected to be unused, not signs of a mapping problem.
    report = WeightConverter(
        model, state_dict, mapper,
        skip_patterns=[r"num_batches_tracked$", r"^fc\."],
    ).convert(strict=False, verbose=True)
    return model, report


def segformer_b0_ade20k(input_shape=(512, 512, 3), cache_dir=None, strict=True):
    """Full SegFormer-B0 (MiT-B0 encoder + all-MLP decode head), pretrained
    and fine-tuned on ADE20K (150 classes) -
    `nvidia/segformer-b0-finetuned-ade-512-512` from HuggingFace. Requires
    `transformers` installed (conversion-time only, to download/read the
    source checkpoint)."""
    from transformers import SegformerForSemanticSegmentation

    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512", cache_dir=cache_dir
    )
    hf_state_dict = {k: v.detach().numpy() for k, v in hf_model.state_dict().items()}
    cfg = MIT_CONFIGS["b0"]
    state_dict = convert_hf_segformer_state_dict(hf_state_dict, cfg)

    model = SegFormer(input_shape=input_shape, num_classes=150, variant="b0")
    model(np.zeros((1,) + input_shape, dtype="float32"))  # build

    mapper = build_segformer_mapper(cfg)
    param_kind_map = build_segformer_param_kind_map(cfg)
    report = WeightConverter(model, state_dict, mapper, param_kind_map=param_kind_map).convert(
        strict=strict, verbose=True
    )
    return model, report


def satmae_vit_base_mae(img_size=224, cache_dir=None, strict=True):
    """ViT-Base MAE encoder + decoder, pretrained on ImageNet-1k - Meta
    AI's official `mae_visualize_vit_base.pth` checkpoint (the "visualize"
    release, which is the one that keeps the decoder for reconstruction
    demos; the plain `mae_pretrain_vit_base.pth` release strips it).
    Architecturally identical to `SatMAEEncoder`/`SatMAEDecoder` with
    `mode="single"` (see module docstring for why this stands in for a
    SatMAE-specific checkpoint)."""
    path = _download(
        "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth",
        "mae_visualize_vit_base.pth", cache_dir,
    )
    encoder = SatMAEEncoder(img_size=img_size, patch_size=16, in_chans=3, embed_dim=768,
                             depth=12, num_heads=12, mode="single")
    decoder = SatMAEDecoder(num_patches=(img_size // 16) ** 2, patch_size=16, in_chans=3,
                             decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                             encoder_embed_dim=768)
    x0 = np.zeros((1, img_size, img_size, 3), dtype="float32")
    tok0, mask0, ids0 = encoder(x0, apply_masking=True, mask_ratio=0.75)
    decoder(tok0, ids0)  # build

    state_dict = load_torch_state_dict_as_numpy(path)
    mapper = build_satmae_mapper()
    # Each converter only owns half of a shared state_dict - skip the
    # other half's keys so `strict=True` checks "did every weight *this*
    # model needs get matched" rather than tripping over the sibling
    # half's keys looking "unused".
    enc_report = WeightConverter(
        encoder, state_dict, mapper,
        skip_patterns=[r"^decoder", r"^mask_token$"],
    ).convert(strict=strict, verbose=True)
    dec_report = WeightConverter(
        decoder, state_dict, mapper,
        skip_patterns=[r"^cls_token$", r"^pos_embed$", r"^patch_embed", r"^blocks\.", r"^norm\.(weight|bias)$"],
    ).convert(strict=strict, verbose=True)
    return encoder, decoder, {"encoder": enc_report, "decoder": dec_report}


def prithvi_eo_100m(img_size=224, cache_dir=None, strict=True):
    """`PrithviEncoder` (embed_dim=768, depth=12, num_heads=12), pretrained
    on NASA HLS imagery - IBM/NASA's official
    `ibm-nasa-geospatial/Prithvi-EO-1.0-100M` checkpoint
    (`Prithvi_EO_V1_100M.pt`, encoder half only)."""
    path = _download(
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/"
        "Prithvi_EO_V1_100M.pt",
        "Prithvi_EO_V1_100M.pt", cache_dir,
    )
    encoder = PrithviEncoder(img_size=img_size, patch_size=16, num_frames=3, tubelet_size=1,
                              in_chans=6, embed_dim=768, depth=12, num_heads=12, name="encoder")
    encoder(np.zeros((1, 3, img_size, img_size, 6), dtype="float32"))  # build

    # The checkpoint is the *full* PrithviMAE (encoder+decoder); the
    # encoder half is nested under an "encoder." prefix matching this
    # `encoder` submodel's own name, so it's stripped then re-added by
    # `build_vit_mapper("encoder")` rather than left in place.
    state_dict = load_torch_state_dict_as_numpy(path, key_prefix_strip="encoder.")
    mapper = build_vit_mapper("encoder")
    report = WeightConverter(
        encoder, state_dict, mapper,
        param_kind_map={"encoder/patch_embed/proj/kernel": "conv3d_kernel"},
        skip_patterns=[r"^decoder"],
    ).convert(strict=strict, verbose=True)
    return encoder, report


def croma_base(img_size=120, cache_dir=None, strict=True):
    """`CROMA(size="base")`, pretrained on paired Sentinel-1/Sentinel-2
    imagery - the paper authors' own released `antofuller/CROMA`
    checkpoint (`CROMA_base.pt`)."""
    return _croma(size="base", img_size=img_size, cache_dir=cache_dir, strict=strict)


def croma_large(img_size=120, cache_dir=None, strict=True):
    """`CROMA(size="large")` - `antofuller/CROMA`'s `CROMA_large.pt`."""
    return _croma(size="large", img_size=img_size, cache_dir=cache_dir, strict=strict)


def _croma(size, img_size, cache_dir, strict):
    filename = f"CROMA_{size}.pt"
    path = _download(f"https://huggingface.co/antofuller/CROMA/resolve/main/{filename}",
                      filename, cache_dir)
    depth = {"base": 12, "large": 24}[size]

    model = CROMA(img_size=img_size, patch_size=8, size=size)
    flat = load_croma_checkpoint(path)
    translated = convert_croma_state_dict(flat, encoder_depth=depth)
    report = WeightConverter(model, translated, build_croma_identity_mapper()).convert(
        strict=strict, verbose=True
    )
    return model, report
