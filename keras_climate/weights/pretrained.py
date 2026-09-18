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
  - `scalemae_vitlarge_fmow`: exact architecture + checkpoint match
    (TorchGeo's clean re-export of the official facebookresearch/scale-mae
    fMoW-RGB ViT-L/16 encoder; the decoder/FPN half of the official full
    checkpoint isn't implemented here - see `remote_sensing/scalemae.py`).
  - `ssl4eo_resnet50_moco`: exact architecture + checkpoint match
    (TorchGeo's clean re-export of the official zhu-xlab/SSL4EO-S12
    MoCo v2 ResNet-50, 13-band Sentinel-2 stem).
  - `satclip_location_encoder_resnet18_l10`: exact architecture + checkpoint
    match (Microsoft's own official `SatCLIP-ResNet18-L10`; only the
    location-encoder half, not the paired image encoder - see
    `remote_sensing/satclip.py`'s module docstring on the reverse-engineered
    spherical-harmonics normalization).
  - `prithvi_eo_v2_300m`: exact architecture + checkpoint match (IBM/NASA's
    official `Prithvi-EO-2.0-300M`, encoder only, base non-"-TL" variant -
    see `foundation/prithvi.py`'s module docstring).
  - `fourcastnet_backbone`: exact architecture + checkpoint match (NVIDIA's
    official FourCastNet backbone, mirrored non-interactively at NERSC -
    see `weather/fourcastnet.py`'s module docstring).
  - `climax_1_40625deg`: exact architecture + checkpoint match (Microsoft's
    official ClimaX checkpoint - see `weather/climax.py`'s module
    docstring and `weights/mappings/climax_mapping.py`'s note on this
    checkpoint's `channel_*` vs the GitHub source's `var_*` naming).
  - `pangu_weather_24`: exact architecture + checkpoint match (a clean
    PyTorch conversion of Huawei's official Pangu-Weather ONNX release -
    **BY-NC-SA 4.0, non-commercial use only** - see
    `weather/pangu_weather.py`'s module docstring on its memory/backend
    requirements and the input-preprocessing scope boundary).

FourCastNet v2 (SFNO) is deliberately not implemented: it replaces AFNO's
planar FFT with a genuine Spherical Harmonic Transform on the sphere,
which is the model's defining feature (not an implementation detail) and
would require reproducing a correctness-critical numerical transform
library (`torch-harmonics`) from scratch to load its real checkpoint
faithfully - out of scope for this pass.

None of the `operators` models (AFNO, FNO, DeepONet, UNO) have a loader
here, despite two of them having a real checkpoint that exists somewhere:
  - AFNO: the only released AFNO weights are FourCastNet's own (already
    covered by `fourcastnet_backbone` above) - no standalone, task-
    agnostic AFNO checkpoint exists.
  - FNO: a real checkpoint exists (`pdebench-fno-audit/fno-weights`), but
    its state_dict reveals a modified block structure (`spectral`/
    `pointwise`/`local_conv`/a scalar `gate`) that doesn't match the
    original paper's code, PDEBench's own repo, or any indexed version of
    `neuraloperator` - its exact combination formula is unverifiable from
    any public source (see `operators/fno.py`'s module docstring).
  - DeepONet: a real checkpoint exists (`BGLab/DeepONet-FlowBench-FPO`),
    but its branch net is preceded by a custom multi-scale Inception-style
    CNN feature extractor whose exact wiring can't be determined from
    tensor shapes/names alone (see `operators/deeponet.py`'s module
    docstring).
  - UNO: no checkpoint exists anywhere (neither the paper's code nor
    `neuraloperator` ships one).
All four are validated only against from-scratch synthetic PyTorch
references of their own architectures (see each module's docstring and
`weights/mappings/{afno,fno,deeponet,uno}_mapping.py`).

Clay and AnySat have no loader here: Clay's official checkpoint is ~5GB
(impractical to fetch/validate in most environments), and AnySat's
official checkpoint uses a substantially more complex, config-driven
architecture than `AnySatEncoder` implements (see
`keras_climate.foundation.anysat`'s module docstring) - both are still
validated against synthetic references matching their respective
mappings' assumed naming (see `foundation/test_clay.py` /
`foundation/test_anysat.py`), just not against the real public checkpoints.

RingMo has no loader either: no official or credible unofficial
checkpoint has ever been publicly released for it at all (see
`remote_sensing/ringmo.py`'s module docstring) - it's validated only
against a from-scratch synthetic reference of this repo's own
architecture.
"""

import os
import urllib.request
import zipfile

import numpy as np

from keras_climate.remote_sensing import UNet, DeepLabV3Plus, SegFormer, MIT_CONFIGS, SatMAEEncoder, SatMAEDecoder
from keras_climate.remote_sensing.scalemae import ScaleMAEEncoder
from keras_climate.remote_sensing.ssl4eo import SSL4EOResNet50
from keras_climate.remote_sensing.satclip import SatCLIPLocationEncoder
from keras_climate.foundation import PrithviEncoder
from keras_climate.foundation.prithvi import PRITHVI_CONFIGS
from keras_climate.foundation.croma import CROMA
from keras_climate.weather.fourcastnet import FourCastNet
from keras_climate.weather.climax import ClimaX
from keras_climate.weather.pangu_weather import PanguWeather
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
    build_scalemae_mapper,
    SCALEMAE_SKIP_PATTERNS,
    build_ssl4eo_mapper,
    load_satclip_checkpoint,
    build_satclip_location_mapper,
    SATCLIP_SKIP_PATTERNS,
    build_fourcastnet_mapper,
    convert_fourcastnet_state_dict,
    load_fourcastnet_checkpoint,
    build_climax_mapper,
    build_pangu_weather_mapper,
    convert_pangu_weather_state_dict,
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


def scalemae_vitlarge_fmow(img_size=224, cache_dir=None, strict=True):
    """`ScaleMAEEncoder` (ViT-L/16, embed_dim=1024, depth=24, heads=16),
    pretrained on fMoW-RGB (800 epochs) - TorchGeo's clean re-export of the
    official facebookresearch/scale-mae checkpoint
    (`vit_large_patch16_224_fmow_rgb_scalemae`), which uses plain timm ViT
    key names (unlike the official full checkpoint, which also bundles the
    Laplacian-pyramid FPN decoder this repo doesn't implement - see module
    docstring)."""
    path = _download(
        "https://huggingface.co/isaaccorley/vit_large_patch16_224_fmow_rgb_scalemae/resolve/main/"
        "vit_large_patch16_224_fmow_rgb_scalemae-386989c9.pth",
        "vit_large_patch16_224_fmow_rgb_scalemae-386989c9.pth", cache_dir,
    )
    encoder = ScaleMAEEncoder(img_size=img_size, patch_size=16, in_chans=3, embed_dim=1024,
                               depth=24, num_heads=16)
    encoder([np.zeros((1, img_size, img_size, 3), dtype="float32"), np.zeros((1,), dtype="float32")])

    state_dict = load_torch_state_dict_as_numpy(path)
    mapper = build_scalemae_mapper()
    # `pos_embed`/`decoder_pos_embed` are dead weight in the source
    # checkpoint (see module docstring) and have no Keras counterpart -
    # skip rather than treat as "unused". Conversely,
    # `pos_embed/{grid_h,grid_w,omega}` are non-trainable buffers on the
    # *Keras* side, derived purely from config (see
    # `GSDPositionalEmbedding`) - they have no source-key counterpart by
    # design, so `strict=True` can only ever mean "every *other* weight
    # matched", not "zero missing".
    report = WeightConverter(
        encoder, state_dict, mapper,
        skip_patterns=SCALEMAE_SKIP_PATTERNS,
    ).convert(strict=False, verbose=True)
    expected_missing = {
        "scalemae_encoder/pos_embed/grid_h",
        "scalemae_encoder/pos_embed/grid_w",
        "scalemae_encoder/pos_embed/omega",
    }
    if strict and (set(report["missing_in_source"]) - expected_missing or report["unused_source_keys"]):
        raise ValueError(
            f"Strict conversion failed: {len(report['missing_in_source'])} missing "
            f"(beyond the expected GSD buffers), {len(report['unused_source_keys'])} unused."
        )
    return encoder, report


def ssl4eo_resnet50_moco(input_shape=(224, 224, 13), cache_dir=None, strict=True):
    """`SSL4EOResNet50`, self-supervised (MoCo v2) pretrained on 13-band
    Sentinel-2 L1C imagery - TorchGeo's clean re-export of the official
    zhu-xlab/SSL4EO-S12 checkpoint (`resnet50_sentinel2_all_moco`; the
    official repo's own release is Google-Drive-hosted and unsuitable for
    non-interactive download). Backbone-only (no `fc` head)."""
    path = _download(
        "https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/"
        "da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth",
        "resnet50_sentinel2_all_moco-df8b932e.pth", cache_dir,
    )
    model = SSL4EOResNet50(input_shape=input_shape)
    model(np.zeros((1,) + input_shape, dtype="float32"))  # build

    state_dict = load_torch_state_dict_as_numpy(path)
    mapper = build_ssl4eo_mapper(layer_counts=(3, 4, 6, 3))
    report = WeightConverter(
        model, state_dict, mapper,
        skip_patterns=[r"num_batches_tracked$"],
    ).convert(strict=strict, verbose=True)
    return model, report


def satclip_location_encoder_resnet18_l10(cache_dir=None, strict=True):
    """`SatCLIPLocationEncoder` (legendre_polys=10, dim_hidden=512,
    num_hidden_layers=2, embed_dim=256), the exact SirenNet weights from
    Microsoft's official `microsoft/SatCLIP-ResNet18-L10` checkpoint (the
    smallest released variant) - only the location-encoder half is ported,
    not the paired ResNet-18 image encoder (see module docstring)."""
    path = _download(
        "https://huggingface.co/microsoft/SatCLIP-ResNet18-L10/resolve/main/satclip-resnet18-l10.ckpt",
        "satclip-resnet18-l10.ckpt", cache_dir,
    )
    model = SatCLIPLocationEncoder(legendre_polys=10, dim_hidden=512, num_hidden_layers=2, embed_dim=256)
    model(np.zeros((1, 2), dtype="float32"))  # build

    state_dict = load_satclip_checkpoint(path)
    mapper = build_satclip_location_mapper(num_hidden_layers=2)
    report = WeightConverter(
        model, state_dict, mapper,
        skip_patterns=SATCLIP_SKIP_PATTERNS,
    ).convert(strict=strict, verbose=True)
    return model, report


def prithvi_eo_v2_300m(img_size=224, num_frames=4, cache_dir=None, strict=True):
    """`PrithviEncoder` at Prithvi-EO-2.0's 300M config (embed_dim=1024,
    depth=24, num_heads=16) - IBM/NASA's official
    `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` checkpoint (encoder half
    only; base non-"-TL" variant, architecturally identical to Prithvi-EO
    -1.0 otherwise - see `foundation/prithvi.py`'s module docstring)."""
    path = _download(
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/"
        "Prithvi_EO_V2_300M.pt",
        "Prithvi_EO_V2_300M.pt", cache_dir,
    )
    cfg = PRITHVI_CONFIGS["prithvi_eo_v2_300m"]
    encoder = PrithviEncoder(img_size=img_size, patch_size=cfg["patch_size"], num_frames=num_frames,
                              tubelet_size=1, in_chans=6, embed_dim=cfg["embed_dim"],
                              depth=cfg["depth"], num_heads=cfg["num_heads"], name="encoder")
    encoder(np.zeros((1, num_frames, img_size, img_size, 6), dtype="float32"))  # build

    # Same "encoder." nesting as Prithvi-EO-1.0's full-MAE checkpoint.
    state_dict = load_torch_state_dict_as_numpy(path, key_prefix_strip="encoder.")
    mapper = build_vit_mapper("encoder")
    report = WeightConverter(
        encoder, state_dict, mapper,
        param_kind_map={"encoder/patch_embed/proj/kernel": "conv3d_kernel"},
        skip_patterns=[r"^decoder"],
    ).convert(strict=strict, verbose=True)
    return encoder, report


def fourcastnet_backbone(cache_dir=None, strict=True):
    """`FourCastNet` at its full released config (720x1440 grid,
    patch_size=8, embed_dim=768, depth=12, 20 ERA5 variables) - NVIDIA's
    official checkpoint, mirrored non-interactively at NERSC (the
    project's own Globus link requires an account). Requires
    `ruamel.yaml` installed in addition to `torch` (see
    `load_fourcastnet_checkpoint`'s docstring) and downloads ~855MB."""
    path = _download(
        "https://portal.nersc.gov/project/m4134/FCN_weights_v0/backbone.ckpt",
        "fourcastnet_backbone.ckpt", cache_dir,
    )
    img_size, patch_size = (720, 1440), 8
    grid_h, grid_w = img_size[0] // patch_size, img_size[1] // patch_size
    model = FourCastNet(img_size=img_size, patch_size=patch_size, in_chans=20, out_chans=20,
                         embed_dim=768, depth=12, num_blocks=8)
    model(np.zeros((1,) + img_size + (20,), dtype="float32"))  # build

    raw_state_dict = load_fourcastnet_checkpoint(path)
    state_dict = convert_fourcastnet_state_dict(raw_state_dict, grid_h, grid_w)
    mapper = build_fourcastnet_mapper(depth=12)
    report = WeightConverter(model, state_dict, mapper).convert(strict=strict, verbose=True)
    return model, report


def climax_1_40625deg(cache_dir=None, strict=True):
    """`ClimaX` at its 1.40625-degree config (128x256 grid, patch_size=4,
    48 variables, embed_dim=1024, depth=8) - Microsoft's official
    checkpoint, hosted directly on HuggingFace (`microsoft/ClimaX`,
    `1.40625deg.ckpt`). Note: the released checkpoint's per-variable-
    aggregation parameters are named `channel_embed`/`channel_query`/
    `channel_agg`, not the `var_*` naming in the current GitHub source -
    see `weights/mappings/climax_mapping.py`'s module docstring."""
    path = _download(
        "https://huggingface.co/microsoft/ClimaX/resolve/main/1.40625deg.ckpt",
        "climax_1_40625deg.ckpt", cache_dir,
    )
    img_size, patch_size, num_vars = (128, 256), 4, 48
    model = ClimaX(img_size=img_size, patch_size=patch_size, num_vars=num_vars,
                    embed_dim=1024, depth=8, decoder_depth=2, num_heads=16)
    model([np.zeros((1,) + img_size + (num_vars,), dtype="float32"),
           np.zeros((1, 1), dtype="float32")])  # build

    state_dict = load_torch_state_dict_as_numpy(path, key_prefix_strip="net.")
    mapper = build_climax_mapper(num_vars=num_vars, depth=8, decoder_depth=2)
    report = WeightConverter(model, state_dict, mapper).convert(strict=strict, verbose=True)
    return model, report


def pangu_weather_24(cache_dir=None, strict=True):
    """`PanguWeather` (24-hour forecast lead time, the full released
    config: 721x1440 grid, dims=(192,384,384,192), depths=(2,6,6,2)) -
    a clean PyTorch conversion of Huawei's official ONNX release, from
    github.com/zhaoshan2/pangu-pytorch's HuggingFace dataset
    (`zhaoshan/pangu_pytorch`, `pretrained_model.zip`, containing both the
    original `.onnx` and the already-converted `.pth` this loader uses).

    **BY-NC-SA 4.0 - non-commercial use only** (the underlying weights are
    Huawei's, regardless of which conversion path produced this specific
    file).

    Building the model and loading these weights works on any Keras
    backend, but actually *calling* the returned model (a full forward
    pass over the real 721x1440x13-level grid) is memory-heavy enough
    that it reliably completes only under the PyTorch backend
    (`KERAS_BACKEND=torch`, with `torch.no_grad()`) in a typical
    development environment - the TensorFlow backend's graph-tracing
    retains enough intermediate state to exhaust memory on a machine with
    tens of GB free, even though this is architecturally the same
    computation either way (see `weather/pangu_weather.py`'s module
    docstring on why the attention step alone needs multiple GB per
    block). This mirrors the real official model's own resource
    requirements (typically run on a GPU with substantial memory).

    Inputs the returned model expects: see `PanguWeather`'s docstring -
    already-normalized, already-concatenated 6-channel upper-air and
    7-channel surface tensors (this loader does not fetch or apply the
    official `aux_data.zip` normalization statistics/masks/constant
    field, which are a separate, non-parameter data-preprocessing
    concern - see `weather/pangu_weather.py`'s module docstring)."""
    zip_path = _download(
        "https://huggingface.co/datasets/zhaoshan/pangu_pytorch/resolve/main/pretrained_model.zip",
        "pangu_pretrained_model.zip", cache_dir,
    )
    extract_dir = os.path.join(os.path.dirname(zip_path), "pangu_pretrained_model")
    pth_path = os.path.join(extract_dir, "pretrained_model", "pangu_weather_24_torch.pth")
    if not os.path.exists(pth_path):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    # `PanguWeather()` is a pure Functional-API `keras.Model` (built from
    # explicit `keras.Input` calls), so its weights already exist right
    # after construction - unlike a subclassed model with a lazy
    # `build()`, no throwaway forward pass is needed just to materialize
    # them. Skipping it matters here specifically: an extra full-
    # resolution forward pass roughly doubles this function's peak/
    # cumulative memory footprint (see module docstring on why a single
    # pass alone already needs several GB).
    model = PanguWeather()

    import torch
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state_dict = {k: v.detach().numpy() for k, v in ckpt["model"].items()}
    state_dict = convert_pangu_weather_state_dict(state_dict)

    mapper = build_pangu_weather_mapper(depths=(2, 6, 6, 2))
    # `attn_mask` buffers are derived, non-trainable constants (see
    # `pangu_weather.py`'s `EarthSpecificBlock`) with no checkpoint
    # counterpart by design - `strict=True` can only mean "every *other*
    # weight matched", not "zero missing".
    report = WeightConverter(model, state_dict, mapper).convert(strict=False, verbose=True)
    expected_missing = {k for k in report["missing_in_source"] if k.endswith("attn_mask")}
    if strict and (set(report["missing_in_source"]) - expected_missing or report["unused_source_keys"]):
        raise ValueError(
            f"Strict conversion failed: {len(report['missing_in_source'])} missing "
            f"(beyond the expected attn_mask buffers), {len(report['unused_source_keys'])} unused."
        )
    return model, report


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
