# Pretrained Weights

`keras_climate.weights.pretrained` provides "timm-style" pretrained-weight
loaders: each function downloads a real, publicly hosted PyTorch
checkpoint (cached locally so it's only fetched once), ports it through
`WeightConverter` + the matching name-mapping, and returns a ready
`keras.Model` (or encoder/decoder pair) with real pretrained weights loaded
— no training required to get a working model.

```python
from keras_climate.weights.pretrained import unet_carvana

model, report = unet_carvana()
model.summary()
```

Every loader returns `(model, report)` (or `(encoder, decoder, report)` for
SatMAE) where `report` is the `WeightConverter.convert(...)` report dict,
so callers can inspect exactly what did/didn't get matched.

## Coverage

Not every architecture in this repo has a publicly downloadable checkpoint
that matches it exactly.

| Loader | Checkpoint | Match |
|---|---|---|
| `unet_carvana` | milesial/Pytorch-UNet's Carvana release | exact architecture + checkpoint |
| `deeplabv3plus_resnet50_imagenet_backbone` | torchvision ImageNet-1k ResNet-50 | backbone only — no public ASPP/decoder checkpoint exists |
| `segformer_b0_ade20k` | `nvidia/segformer-b0-finetuned-ade-512-512` (HF) | exact — requires `transformers` |
| `satmae_vit_base_mae` | Meta AI's official ImageNet-1k MAE ViT-Base | stand-in — SatMAE's own checkpoints aren't hosted; `mode="single"` is architecturally identical to plain ViT-MAE |
| `prithvi_eo_100m` | `ibm-nasa-geospatial/Prithvi-EO-1.0-100M` | exact — encoder only |
| `croma_base` / `croma_large` | `antofuller/CROMA` (`CROMA_base.pt` / `CROMA_large.pt`) | exact — both modalities + joint fusion |
| `scalemae_vitlarge_fmow` | TorchGeo re-export of facebookresearch/scale-mae fMoW-RGB ViT-L/16 | exact — decoder/FPN not implemented |
| `ssl4eo_resnet50_moco` | TorchGeo re-export of zhu-xlab/SSL4EO-S12 MoCo v2 ResNet-50 | exact |
| `satclip_location_encoder_resnet18_l10` | Microsoft's official `SatCLIP-ResNet18-L10` | exact — location encoder only, not the paired image encoder |
| `prithvi_eo_v2_300m` | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` | exact — encoder only, base non-"-TL" variant |
| `fourcastnet_backbone` | NVIDIA's official release, mirrored at NERSC | exact — requires `ruamel.yaml` + `torch` |
| `climax_1_40625deg` | Microsoft's official `microsoft/ClimaX` (`1.40625deg.ckpt`) | exact |
| `pangu_weather_24` | Clean PyTorch conversion of Huawei's ONNX release | exact — **BY-NC-SA 4.0, non-commercial use only** |
| `anysat_base` | `g-astruc/AnySat` (HuggingFace), "base" config | exact — validated end-to-end to ~1e-6 against real PyTorch forward passes |

## No loader available

- **AFNO** — the only released AFNO weights are FourCastNet's own (already
  covered by `fourcastnet_backbone`); no standalone, task-agnostic AFNO
  checkpoint exists.
- **FNO** — a real checkpoint exists (`pdebench-fno-audit/fno-weights`),
  but its state_dict reveals a modified block structure that doesn't match
  the original paper's code, PDEBench's own repo, or any indexed version of
  `neuraloperator` — its exact combination formula is unverifiable from any
  public source.
- **DeepONet** — a real checkpoint exists (`BGLab/DeepONet-FlowBench-FPO`),
  but its branch net is preceded by a custom multi-scale Inception-style
  CNN feature extractor whose exact wiring can't be determined from tensor
  shapes/names alone.
- **UNO** — no checkpoint exists anywhere (neither the paper's code nor
  `neuraloperator` ships one).
- **Clay** — the official checkpoint is ~5GB, impractical to fetch/validate
  in most environments; still validated against a synthetic reference
  matching the assumed mapping naming, just not the real public checkpoint.
- **`AnySatEncoder`** (this repo's own simpler AnySat design) — validated
  only against a from-scratch synthetic reference, not the real checkpoint
  (see `anysat_base`/`AnySatRelease` above for the faithful port that *is*
  checkpoint-compatible).
- **RingMo** — no official or credible unofficial checkpoint has ever been
  publicly released for it at all.

All unloaded models are validated only against from-scratch synthetic
PyTorch references of their own architectures — see
[Weight Porting](weight-porting.md).

## Out of scope

**FourCastNet v2 (SFNO)** is deliberately not implemented: it replaces
AFNO's planar FFT with a genuine Spherical Harmonic Transform on the
sphere, which is the model's defining feature (not an implementation
detail) and would require reproducing a correctness-critical numerical
transform library (`torch-harmonics`) from scratch to load its real
checkpoint faithfully.
