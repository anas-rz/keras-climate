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
from keras_climate.foundation.anysat_release import AnySatRelease, ANYSAT_MODALITIES, anysat_projector_configs
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
    build_anysat_release_mapper,
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
    path = _download(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/"
        "unet_carvana_scale1.0_epoch2.pth",
        "unet_carvana_scale1.0_epoch2.pth", cache_dir,
    )
    model = UNet(input_shape=input_shape, num_classes=2, base_filters=64, depth=4)
    model(np.zeros((1,) + input_shape, dtype="float32"))

    state_dict = load_torch_state_dict_as_numpy(path)
    report = WeightConverter(
        model, state_dict, build_unet_mapper(depth=4),
        skip_patterns=[r"num_batches_tracked$"],
    ).convert(strict=strict, verbose=True)
    return model, report


def deeplabv3plus_resnet50_imagenet_backbone(input_shape=(512, 512, 3), num_classes=21, cache_dir=None):
    path = _download(
        "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "resnet50-11ad3fa6.pth", cache_dir,
    )
    model = DeepLabV3Plus(input_shape=input_shape, num_classes=num_classes,
                           backbone_layers=(3, 4, 6, 3), output_stride=16)
    model(np.zeros((1,) + input_shape, dtype="float32"))

    state_dict = load_torch_state_dict_as_numpy(path)
    mapper = build_resnet_backbone_mapper(layer_counts=(3, 4, 6, 3))
    report = WeightConverter(
        model, state_dict, mapper,
        skip_patterns=[r"num_batches_tracked$", r"^fc\."],
    ).convert(strict=False, verbose=True)
    return model, report


def segformer_b0_ade20k(input_shape=(512, 512, 3), cache_dir=None, strict=True):
    from transformers import SegformerForSemanticSegmentation

    hf_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512", cache_dir=cache_dir
    )
    hf_state_dict = {k: v.detach().numpy() for k, v in hf_model.state_dict().items()}
    cfg = MIT_CONFIGS["b0"]
    state_dict = convert_hf_segformer_state_dict(hf_state_dict, cfg)

    model = SegFormer(input_shape=input_shape, num_classes=150, variant="b0")
    model(np.zeros((1,) + input_shape, dtype="float32"))

    mapper = build_segformer_mapper(cfg)
    param_kind_map = build_segformer_param_kind_map(cfg)
    report = WeightConverter(model, state_dict, mapper, param_kind_map=param_kind_map).convert(
        strict=strict, verbose=True
    )
    return model, report


def satmae_vit_base_mae(img_size=224, cache_dir=None, strict=True):
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
    decoder(tok0, ids0)

    state_dict = load_torch_state_dict_as_numpy(path)
    mapper = build_satmae_mapper()
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
    path = _download(
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/"
        "Prithvi_EO_V1_100M.pt",
        "Prithvi_EO_V1_100M.pt", cache_dir,
    )
    encoder = PrithviEncoder(img_size=img_size, patch_size=16, num_frames=3, tubelet_size=1,
                              in_chans=6, embed_dim=768, depth=12, num_heads=12, name="encoder")
    encoder(np.zeros((1, 3, img_size, img_size, 6), dtype="float32"))

    state_dict = load_torch_state_dict_as_numpy(path, key_prefix_strip="encoder.")
    mapper = build_vit_mapper("encoder")
    report = WeightConverter(
        encoder, state_dict, mapper,
        param_kind_map={"encoder/patch_embed/proj/kernel": "conv3d_kernel"},
        skip_patterns=[r"^decoder"],
    ).convert(strict=strict, verbose=True)
    return encoder, report


def scalemae_vitlarge_fmow(img_size=224, cache_dir=None, strict=True):
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
    path = _download(
        "https://hf.co/torchgeo/resnet50_sentinel2_all_moco/resolve/"
        "da4f3c9dbe09272eb902f3b37f46635fa4726879/resnet50_sentinel2_all_moco-df8b932e.pth",
        "resnet50_sentinel2_all_moco-df8b932e.pth", cache_dir,
    )
    model = SSL4EOResNet50(input_shape=input_shape)
    model(np.zeros((1,) + input_shape, dtype="float32"))

    state_dict = load_torch_state_dict_as_numpy(path)
    mapper = build_ssl4eo_mapper(layer_counts=(3, 4, 6, 3))
    report = WeightConverter(
        model, state_dict, mapper,
        skip_patterns=[r"num_batches_tracked$"],
    ).convert(strict=strict, verbose=True)
    return model, report


def satclip_location_encoder_resnet18_l10(cache_dir=None, strict=True):
    path = _download(
        "https://huggingface.co/microsoft/SatCLIP-ResNet18-L10/resolve/main/satclip-resnet18-l10.ckpt",
        "satclip-resnet18-l10.ckpt", cache_dir,
    )
    model = SatCLIPLocationEncoder(legendre_polys=10, dim_hidden=512, num_hidden_layers=2, embed_dim=256)
    model(np.zeros((1, 2), dtype="float32"))

    state_dict = load_satclip_checkpoint(path)
    mapper = build_satclip_location_mapper(num_hidden_layers=2)
    report = WeightConverter(
        model, state_dict, mapper,
        skip_patterns=SATCLIP_SKIP_PATTERNS,
    ).convert(strict=strict, verbose=True)
    return model, report


def prithvi_eo_v2_300m(img_size=224, num_frames=4, cache_dir=None, strict=True):
    path = _download(
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/"
        "Prithvi_EO_V2_300M.pt",
        "Prithvi_EO_V2_300M.pt", cache_dir,
    )
    cfg = PRITHVI_CONFIGS["prithvi_eo_v2_300m"]
    encoder = PrithviEncoder(img_size=img_size, patch_size=cfg["patch_size"], num_frames=num_frames,
                              tubelet_size=1, in_chans=6, embed_dim=cfg["embed_dim"],
                              depth=cfg["depth"], num_heads=cfg["num_heads"], name="encoder")
    encoder(np.zeros((1, num_frames, img_size, img_size, 6), dtype="float32"))

    state_dict = load_torch_state_dict_as_numpy(path, key_prefix_strip="encoder.")
    mapper = build_vit_mapper("encoder")
    report = WeightConverter(
        encoder, state_dict, mapper,
        param_kind_map={"encoder/patch_embed/proj/kernel": "conv3d_kernel"},
        skip_patterns=[r"^decoder"],
    ).convert(strict=strict, verbose=True)
    return encoder, report


def fourcastnet_backbone(cache_dir=None, strict=True):
    path = _download(
        "https://portal.nersc.gov/project/m4134/FCN_weights_v0/backbone.ckpt",
        "fourcastnet_backbone.ckpt", cache_dir,
    )
    img_size, patch_size = (720, 1440), 8
    grid_h, grid_w = img_size[0] // patch_size, img_size[1] // patch_size
    model = FourCastNet(img_size=img_size, patch_size=patch_size, in_chans=20, out_chans=20,
                         embed_dim=768, depth=12, num_blocks=8)
    model(np.zeros((1,) + img_size + (20,), dtype="float32"))

    raw_state_dict = load_fourcastnet_checkpoint(path)
    state_dict = convert_fourcastnet_state_dict(raw_state_dict, grid_h, grid_w)
    mapper = build_fourcastnet_mapper(depth=12)
    report = WeightConverter(model, state_dict, mapper).convert(strict=strict, verbose=True)
    return model, report


def climax_1_40625deg(cache_dir=None, strict=True):
    path = _download(
        "https://huggingface.co/microsoft/ClimaX/resolve/main/1.40625deg.ckpt",
        "climax_1_40625deg.ckpt", cache_dir,
    )
    img_size, patch_size, num_vars = (128, 256), 4, 48
    model = ClimaX(img_size=img_size, patch_size=patch_size, num_vars=num_vars,
                    embed_dim=1024, depth=8, decoder_depth=2, num_heads=16)
    model([np.zeros((1,) + img_size + (num_vars,), dtype="float32"),
           np.zeros((1, 1), dtype="float32")])

    state_dict = load_torch_state_dict_as_numpy(path, key_prefix_strip="net.")
    mapper = build_climax_mapper(num_vars=num_vars, depth=8, decoder_depth=2)
    report = WeightConverter(model, state_dict, mapper).convert(strict=strict, verbose=True)
    return model, report


def pangu_weather_24(cache_dir=None, strict=True):
    zip_path = _download(
        "https://huggingface.co/datasets/zhaoshan/pangu_pytorch/resolve/main/pretrained_model.zip",
        "pangu_pretrained_model.zip", cache_dir,
    )
    extract_dir = os.path.join(os.path.dirname(zip_path), "pangu_pretrained_model")
    pth_path = os.path.join(extract_dir, "pretrained_model", "pangu_weather_24_torch.pth")
    if not os.path.exists(pth_path):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    model = PanguWeather()

    import torch
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state_dict = {k: v.detach().numpy() for k, v in ckpt["model"].items()}
    state_dict = convert_pangu_weather_state_dict(state_dict)

    mapper = build_pangu_weather_mapper(depths=(2, 6, 6, 2))
    report = WeightConverter(model, state_dict, mapper).convert(strict=False, verbose=True)
    expected_missing = {k for k in report["missing_in_source"] if k.endswith("attn_mask")}
    if strict and (set(report["missing_in_source"]) - expected_missing or report["unused_source_keys"]):
        raise ValueError(
            f"Strict conversion failed: {len(report['missing_in_source'])} missing "
            f"(beyond the expected attn_mask buffers), {len(report['unused_source_keys'])} unused."
        )
    return model, report


def croma_base(img_size=120, cache_dir=None, strict=True):
    return _croma(size="base", img_size=img_size, cache_dir=cache_dir, strict=strict)


def croma_large(img_size=120, cache_dir=None, strict=True):
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


def anysat_base(modalities=None, input_shapes=None, scale=1, cache_dir=None, strict=False):
    path = _download(
        "https://huggingface.co/g-astruc/AnySat/resolve/main/models/AnySat.pth",
        "AnySat.pth", cache_dir,
    )
    if modalities is None:
        modalities = ["aerial", "s2"]

    proj_cfgs = anysat_projector_configs(768)
    if input_shapes is None:
        input_shapes = {}
        for m in modalities:
            pc = proj_cfgs[m]
            if pc["kind"] == "image":
                res = int(10 / pc["resolution"])
                gs = res // pc["patch_size"]
                side = gs * scale * pc["patch_size"]
                input_shapes[m] = (side, side, pc["in_chans"])
            else:
                se = max(1, scale // pc["reduce_scale"])
                input_shapes[m] = (8, se, se, pc["in_channels"])

    model = AnySatRelease(modalities, input_shapes, scale, size="base")
    dummy = {}
    for m in modalities:
        shp = input_shapes[m]
        if proj_cfgs[m]["kind"] == "image":
            dummy[m] = np.zeros((1,) + shp, dtype="float32")
        else:
            dummy[m] = (np.zeros((1,) + shp, dtype="float32"), np.zeros((1, shp[0]), dtype="float32"))
    model(dummy)

    state_dict = load_torch_state_dict_as_numpy(path)
    mapper = build_anysat_release_mapper(modalities, depth=6)
    report = WeightConverter(
        model, state_dict, mapper, skip_patterns=[r"pad_parameter$"],
    ).convert(strict=False, verbose=True)
    expected_missing = {w for w in report["missing_in_source"] if w.endswith("/pe_denom")}
    if strict and (set(report["missing_in_source"]) - expected_missing or report["unused_source_keys"]):
        raise ValueError(
            f"Strict conversion failed: {len(report['missing_in_source'])} missing "
            f"(beyond the expected pe_denom buffers), {len(report['unused_source_keys'])} unused "
            f"(unused is expected/nonzero when `modalities` is a subset of ANYSAT_MODALITIES)."
        )
    return model, report
