# Remote Sensing

`keras_climate.remote_sensing` — pixel-wise segmentation and classification
of aerial/satellite imagery.

## UNet

Classic UNet (Ronneberger et al. 2015) for pixel-wise segmentation of
remote-sensing rasters (land-cover, cloud masks, flood extent, etc).
Supports an arbitrary number of input bands, not just RGB.

A real pretrained checkpoint is available: `unet_carvana` in
[Pretrained Weights](../pretrained-weights.md) (milesial/Pytorch-UNet's
released Carvana weights, exact architecture + checkpoint match).

## DeepLabV3+

Dilated-ResNet backbone + ASPP + a light decoder with a low-level-feature
skip connection (Chen et al. 2018). Commonly used for land-cover
classification and semantic segmentation of aerial/satellite imagery. The
ResNet backbone uses atrous convolutions in its last stage(s) so the
overall output stride matches DeepLab's requirements (8 or 16);
`output_stride=32` gives the standard, non-dilated torchvision ResNet
stride pattern instead, used for plain classification backbones (e.g.
SSL4EO) rather than DeepLab's dense-prediction one.

Only the ResNet-50 backbone has a matching public checkpoint (ImageNet-1k,
via torchvision) — see `deeplabv3plus_resnet50_imagenet_backbone` in
[Pretrained Weights](../pretrained-weights.md). No public checkpoint exists
for the ASPP + decoder head, which stays randomly initialized.

## SegFormer

Hierarchical Mix Transformer (MiT) encoder (Xie et al. 2021) with
efficient spatial-reduction attention + Mix-FFN, and a lightweight all-MLP
decode head. A strong general-purpose choice for satellite/aerial
segmentation.

Real pretrained checkpoint: `segformer_b0_ade20k` — the full encoder +
decode head, fine-tuned on ADE20K (150 classes), ported from
`nvidia/segformer-b0-finetuned-ade-512-512` on HuggingFace. Requires
`transformers` installed at conversion time.

## SatMAE

A Masked Autoencoder ViT (Cong et al. 2022) specialized for satellite
imagery, with two variants:

- **temporal** — groups multi-date image stacks and adds temporal position
  encodings (for Sentinel/Planet time series).
- **multispectral** — encodes each spectral group (e.g. RGB / NIR / SWIR)
  with its own positional + channel embedding before merging into one ViT
  token sequence.

Both variants share the same ViT-MAE encoder/decoder backbone. Use
`SatMAEEncoder(...)` alone and attach your own head for downstream tasks;
use `SatMAE(...)` for MAE pretraining.

SatMAE's own checkpoints aren't hosted for direct download. `mode="single"`
is architecturally identical to a plain ViT-MAE, so `satmae_vit_base_mae`
loads Meta AI's official ImageNet-1k MAE checkpoint as the best publicly
available stand-in for validating the shared core end-to-end.

## Scale-MAE

A Masked Autoencoder ViT (Reed et al. 2023) whose positional embedding is a
function of each image's ground-sample-distance (GSD, in meters/pixel)
rather than a fixed learned/sin-cos table. This makes the same pretrained
encoder usable across satellite/aerial imagery captured at different
physical resolutions — the pixel grid stays the same size but the *scale*
it represents is baked into the positional encoding.

Only the encoder (`ScaleMAEEncoder`) is implemented, for downstream feature
extraction — the official pretraining pipeline's Laplacian-pyramid FPN
decoder (used only to compute the MAE reconstruction loss) is out of scope
for a feature-extraction library.

Real checkpoint: `scalemae_vitlarge_fmow` — TorchGeo's clean re-export of
the official fMoW-RGB ViT-L/16 encoder (800 epochs).

## RingMo

A Swin-Transformer-based masked-image-modeling foundation model (Sun et al.
2022) built around "PI-Mask" (Patch Incomplete Mask): only a fraction of
pixels within each masked block are zeroed, rather than the whole block,
better preserving partial texture/edges of the small, dense objects common
in aerial/satellite imagery than vanilla MAE/SimMIM block masking.

No official or credible unofficial checkpoint has ever been publicly
released for RingMo. This module is validated only against a from-scratch
synthetic PyTorch reference of its own architecture; the patch-embed stem
("pi_conv", a factorized multi-stage conv) is this repo's own reasonable
interpretation of the paper's description, since it isn't documented
precisely enough in public sources to reproduce byte-exact.

## SSL4EO-S12

Not a novel architecture, but a large-scale Sentinel-1/Sentinel-2 dataset
(Wang et al. 2022) paired with several self-supervised pretrained
backbones. `SSL4EOResNet50` ports the released ResNet-50 backbone — a
standard torchvision ResNet-50 with its stem's first conv widened from 3 to
13 input channels (Sentinel-2 L1C's full band set), otherwise
architecturally unchanged.

Real checkpoint: `ssl4eo_resnet50_moco` — TorchGeo's clean re-export of the
official MoCo v2 ResNet-50 (the official repo's own release is
Google-Drive-hosted and unsuitable for non-interactive download).

## SatCLIP (location encoder)

A CLIP-style dual encoder (Klemmer et al. 2023, Microsoft), contrastively
trained so a (lon, lat) coordinate's embedding predicts the embedding of
co-located Sentinel-2 imagery. Only the **location encoder** is
implemented: a real spherical-harmonics positional encoding of the
coordinate, followed by a SirenNet (sinusoidal-activation MLP). It has no
image input, is tiny (~1MB of the checkpoint's 57MB), and is directly
useful standalone for geospatial ML.

Real checkpoint: `satclip_location_encoder_resnet18_l10` — Microsoft's
official `SatCLIP-ResNet18-L10` release (smallest variant); only the
location-encoder half is ported, not the paired image encoder.
