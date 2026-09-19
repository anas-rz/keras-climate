# Weight Porting

The weight-porting framework in `keras_climate.weights` is *generic*: you
describe how PyTorch checkpoint keys map to Keras weight names (either as
regex rules or a Python callable), and `WeightConverter` handles shape
inference (Dense/Conv kernel transposition), assignment, and reports any
weights it could not match on either side.

## How it works

Rather than writing one bespoke script per model that hardcodes every layer
name, you provide:

1. a `name_map`: a function `torch_key -> keras_weight_name | None` (or a
   list of `(regex_pattern, replacement)` rules — see `build_regex_mapper`);
2. optionally, per-parameter-kind transpose rules (Dense/Conv weights are
   laid out differently in PyTorch vs. Keras — handled automatically by
   `infer_transpose` for the common cases, with overrides available).

`WeightConverter.convert()` then walks every trainable/non-trainable weight
in the target Keras model, looks up its source tensor via the name map,
transposes it into Keras layout, checks the shape matches, and assigns it —
reporting anything it could not match on either side, rather than silently
returning a half-initialized model.

```python
from keras_climate.remote_sensing import UNet
from keras_climate.weights import WeightConverter, load_torch_state_dict_as_numpy
from keras_climate.weights.mappings import build_unet_mapper

model = UNet(input_shape=(256, 256, 3), num_classes=1)
model(keras.ops.zeros((1, 256, 256, 3)))  # build the model

state_dict = load_torch_state_dict_as_numpy("unet_carvana.pth")
converter = WeightConverter(model, state_dict, build_unet_mapper())
report = converter.convert(strict=False)

model.save_weights("unet_keras.weights.h5")
```

`infer_transpose` handles the common cases automatically:

| Source (PyTorch)                          | Target (Keras)                    |
| ------------------------------------------ | ---------------------------------- |
| `nn.Linear.weight` `(out, in)`             | `Dense.kernel` `(in, out)`         |
| `nn.Conv2d.weight` `(out, in, kh, kw)`     | `Conv2D.kernel` `(kh, kw, in, out)`|
| `nn.Conv3d.weight` `(out, in, kt, kh, kw)` | `Conv3D.kernel` `(kt, kh, kw, in, out)` |
| `nn.Conv1d.weight` `(out, in, k)`          | `Conv1D.kernel` `(k, in, out)`     |
| depthwise conv `(out=in*mult, 1, kh, kw)`  | `DepthwiseConv2D.kernel` `(kh, kw, in, mult)` |

Biases, norm gamma/beta, embeddings, and 1D vectors are assumed to already
match and pass through unchanged.

## CLI wrapper

`weights/port_weights.py` bundles a model-build function + name-mapper per
checkpoint family:

```bash
python -m keras_climate.weights.port_weights \
    --model unet --checkpoint unet_carvana.pth --output unet_keras.weights.h5
```

Each entry in `MODEL_REGISTRY` wires together: how to build the target
Keras model, how to load the source checkpoint, which name-mapping to use,
and any per-weight `param_kind` overrides that shape inference can't
resolve automatically.

## Adding a new mapping

1. Get the source `state_dict` key names: `list(state_dict.keys())`.
2. Get the target Keras weight names: `[w.path for w in model.weights]`.
3. Write regex rules (see `weights/mappings/unet_mapping.py` for a
   hand-written example, or `weights/mappings/vit_mapping.py` for a
   reusable generic-ViT-block mapper used by every transformer model in
   this repo).
4. Run `converter.convert(strict=False)` first and inspect the "no source
   match" / "unused source keys" report to iterate on the rules; switch to
   `strict=True` once the report is clean.
5. Register the `(build_fn, name_map_fn, loader)` triple in
   `MODEL_REGISTRY` inside `port_weights.py` so it's usable from the CLI.

!!! note
    Exact checkpoint key names vary by which reference repo/checkpoint
    you're porting from (e.g. the official author repo vs. a HuggingFace
    re-upload vs. an `mmsegmentation` config) — the mappings shipped in
    this repo are worked examples of the *pattern*, not guaranteed to match
    every public checkpoint byte-for-byte without adjustment.

## Models without a pretrained checkpoint

Many models in this repo have no publicly available general-purpose
checkpoint at all. For those, the round-trip test suite instead builds a
from-scratch synthetic PyTorch reference implementing the *same*
architecture as the Keras port, and validates `WeightConverter` +
that model's mapper against it — proving the porting infrastructure and
the mapping rules are correct, even without real weights to load. See each
model's page under [Models](models/remote-sensing.md) for which checkpoint
tier it falls into, and [Pretrained Weights](pretrained-weights.md) for the
full picture across the whole repo.
