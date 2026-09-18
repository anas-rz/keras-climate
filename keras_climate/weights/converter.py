"""
keras_climate.weights.converter
-----------------------------------
A generic, model-agnostic framework for porting pretrained weights (almost
always PyTorch `state_dict`s, since that's what every reference
implementation in this repo's model list ships) onto the equivalent
`keras_climate` Keras model.

The core idea: rather than writing one bespoke script per model that
hardcodes every layer name, you provide

  1. a `name_map`: a function `torch_key -> keras_weight_name | None` (or a
     list of `(regex_pattern, replacement)` rules - see `build_regex_mapper`),
  2. optionally, per-parameter-kind transpose rules (Dense/Conv weights are
     laid out differently in PyTorch vs Keras - handled automatically by
     `infer_transpose` for the common cases, with overrides available).

`WeightConverter.convert()` then walks every trainable/non-trainable weight
in the target Keras model, looks up its source tensor via the name map,
transposes it into Keras layout, checks the shape matches, and assigns it -
reporting anything it could not match on either side so you can fix the
mapping rather than silently getting a half-initialized model.
"""

import re
import numpy as np


# --------------------------------------------------------------------------
# Layout conversion helpers
# --------------------------------------------------------------------------

def infer_transpose(torch_shape, keras_shape, param_kind=None):
    """Given a source (PyTorch) array shape and the target Keras weight
    shape, return the axis permutation needed to convert, or None if no
    transpose is needed / shapes already match.

    Handles the common cases:
      - nn.Linear.weight  (out, in)              -> Dense.kernel (in, out)
      - nn.Conv2d.weight  (out, in, kh, kw)       -> Conv2D.kernel (kh, kw, in, out)
      - nn.Conv3d.weight  (out, in, kt, kh, kw)   -> Conv3D.kernel (kt, kh, kw, in, out)
      - nn.Conv1d.weight  (out, in, k)            -> Conv1D.kernel (k, in, out)
      - depthwise conv (out=in*mult, 1, kh, kw)   -> DepthwiseConv2D.kernel (kh, kw, in, mult)
    Everything else (biases, norm gamma/beta, embeddings, 1D vectors) is
    assumed to already match and passes through unchanged.

    Note: an explicit `param_kind` is always honored *before* the "shapes
    already match" shortcut below - a square `nn.Linear.weight` (out ==
    in) has the exact same shape as its Keras `Dense.kernel` counterpart
    while still needing the (out, in) -> (in, out) transpose, so shape
    equality alone cannot be trusted to mean "no transform needed" for a
    known dense kernel.
    """
    if param_kind == "dense_kernel":
        return (1, 0)

    if tuple(torch_shape) == tuple(keras_shape):
        return None

    if len(torch_shape) == 2 and torch_shape[::-1] == tuple(keras_shape):
        return (1, 0)

    if param_kind == "conv2d_kernel" or len(torch_shape) == 4:
        # (out, in, kh, kw) -> (kh, kw, in, out)
        candidate = (torch_shape[2], torch_shape[3], torch_shape[1], torch_shape[0])
        if candidate == tuple(keras_shape):
            return (2, 3, 1, 0)

    if param_kind == "conv3d_kernel" or len(torch_shape) == 5:
        # (out, in, kt, kh, kw) -> (kt, kh, kw, in, out)
        candidate = (torch_shape[2], torch_shape[3], torch_shape[4], torch_shape[1], torch_shape[0])
        if candidate == tuple(keras_shape):
            return (2, 3, 4, 1, 0)

    if param_kind == "conv1d_kernel" or len(torch_shape) == 3:
        # (out, in, k) -> (k, in, out)
        candidate = (torch_shape[2], torch_shape[1], torch_shape[0])
        if candidate == tuple(keras_shape):
            return (2, 1, 0)

    if param_kind == "depthwise_kernel" and len(torch_shape) == 4:
        # torch depthwise: (in*mult, 1, kh, kw) -> keras (kh, kw, in, mult)
        return "depthwise"

    return None


def apply_transpose(array, perm):
    if perm is None:
        return array
    if perm == "depthwise":
        out_ch, _, kh, kw = array.shape
        # Assumes multiplier=1 (out_ch == in_ch); generalize if your model uses depth_multiplier > 1.
        # Keras `DepthwiseConv2D.kernel` is (kh, kw, in_channels, multiplier);
        # with multiplier=1, in_channels == out_ch, so the channel axis goes
        # third, not fourth.
        arr = array.reshape(out_ch, 1, kh, kw)
        arr = np.transpose(arr, (2, 3, 0, 1))  # (kh, kw, out_ch, 1)
        return arr
    return np.transpose(array, perm)


# --------------------------------------------------------------------------
# Name-mapping helpers
# --------------------------------------------------------------------------

def build_regex_mapper(rules):
    """`rules`: list of (pattern, replacement) applied with re.sub, tried in
    order; the first pattern that matches (re.search) is used. Returns a
    `torch_key -> keras_key` function; unmatched keys map to None."""

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, replacement in compiled:
            if pattern.search(torch_key):
                return pattern.sub(replacement, torch_key)
        return None

    return mapper


# --------------------------------------------------------------------------
# Converter
# --------------------------------------------------------------------------

class WeightConverter:
    def __init__(self, keras_model, torch_state_dict, name_map, param_kind_map=None,
                 skip_patterns=None):
        """
        Args:
            keras_model: the target `keras.Model` (already built, e.g. by
                calling it once on dummy input, or via `model.build(...)`).
            torch_state_dict: dict[str, np.ndarray] - a PyTorch state_dict
                already converted to numpy (e.g. `{k: v.numpy() for k, v in
                torch_model.state_dict().items()}`).
            name_map: either a `torch_key -> keras_key | None` callable, or
                a list of (regex, replacement) rules (passed to
                `build_regex_mapper`).
            param_kind_map: optional `keras_weight_name -> param_kind`
                override dict for ambiguous shapes (e.g. distinguishing a
                depthwise kernel from a small regular conv kernel). Values:
                "dense_kernel", "conv1d_kernel", "conv2d_kernel",
                "conv3d_kernel", "depthwise_kernel".
            skip_patterns: list of regexes; matching torch keys are ignored
                entirely (e.g. optimizer state, buffers you don't need).
        """
        self.model = keras_model
        self.state_dict = torch_state_dict
        self.name_map = name_map if callable(name_map) else build_regex_mapper(name_map)
        self.param_kind_map = param_kind_map or {}
        self.skip_patterns = [re.compile(p) for p in (skip_patterns or [])]

    def convert(self, strict=False, verbose=True):
        """Applies the mapping and returns a report dict with
        'matched', 'missing_in_source', 'unused_source_keys'."""
        keras_by_name = {w.path: w for w in self.model.weights}
        # `keras.Variable.path` is the fully-qualified name in Keras 3;
        # fall back to `.name` for older Keras if needed.
        if not keras_by_name:
            keras_by_name = {w.name: w for w in self.model.weights}

        # Build torch_key -> keras_key using name_map, skipping ignored keys.
        matched, missing, used_source_keys = [], [], set()

        for keras_key, keras_var in keras_by_name.items():
            torch_key = self._find_source_key(keras_key)
            if torch_key is None:
                missing.append(keras_key)
                continue

            source = np.asarray(self.state_dict[torch_key])
            # A very common real-world checkpoint quirk: a positional/token
            # embedding saved with an extra leading batch-broadcast axis of
            # size 1 (e.g. official MAE's `pos_embed` is (1, N, D)) where
            # this repo's Keras weight is stored without it ((N, D)) and
            # adds the axis back at use time. Squeeze it before shape
            # inference so it doesn't get treated as a hard mismatch.
            if source.ndim == len(keras_var.shape) + 1 and source.shape[0] == 1:
                source = source[0]
            param_kind = self.param_kind_map.get(keras_key)
            if param_kind is None and len(source.shape) == 2 and keras_key.endswith("/kernel"):
                # Keras always names a Dense/Conv*'s weight "kernel" - a
                # directly-added array (pos_embed, cls_token, ...) never
                # is - so this is an unambiguous signal that `source` is an
                # `nn.Linear.weight` needing the (out, in) -> (in, out)
                # transpose, even when it's square and therefore looks
                # like it "already matches" by shape alone (see
                # `infer_transpose`'s docstring for why shape equality
                # can't be trusted for square Dense kernels).
                param_kind = "dense_kernel"
            perm = infer_transpose(source.shape, tuple(keras_var.shape), param_kind)
            converted = apply_transpose(np.asarray(source), perm)

            if tuple(converted.shape) != tuple(keras_var.shape):
                msg = (f"Shape mismatch for '{keras_key}' <- '{torch_key}': "
                       f"source {source.shape} -> converted {converted.shape}, "
                       f"expected {tuple(keras_var.shape)}")
                if strict:
                    raise ValueError(msg)
                if verbose:
                    print(f"[SKIP] {msg}")
                missing.append(keras_key)
                continue

            keras_var.assign(converted)
            matched.append((keras_key, torch_key))
            used_source_keys.add(torch_key)

        unused_source_keys = [k for k in self.state_dict if k not in used_source_keys
                               and not any(p.search(k) for p in self.skip_patterns)]

        if verbose:
            print(f"[keras_climate] converted {len(matched)}/{len(keras_by_name)} weights.")
            if missing:
                print(f"[keras_climate] {len(missing)} Keras weights had no source match:")
                for m in missing[:20]:
                    print(f"    - {m}")
                if len(missing) > 20:
                    print(f"    ... and {len(missing) - 20} more")
            if unused_source_keys:
                print(f"[keras_climate] {len(unused_source_keys)} source keys were unused:")
                for u in unused_source_keys[:20]:
                    print(f"    - {u}")
                if len(unused_source_keys) > 20:
                    print(f"    ... and {len(unused_source_keys) - 20} more")

        if strict and (missing or unused_source_keys):
            raise ValueError(
                f"Strict conversion failed: {len(missing)} missing, "
                f"{len(unused_source_keys)} unused source keys."
            )

        return {
            "matched": matched,
            "missing_in_source": missing,
            "unused_source_keys": unused_source_keys,
        }

    def _find_source_key(self, keras_key):
        # Try direct call first (regex mapper / callable both supported).
        direct = self.name_map(keras_key)
        if direct is not None and direct in self.state_dict:
            return direct

        # Fall back: some mapping functions are defined torch_key -> keras_key
        # (forward direction); search for a torch key whose mapped value
        # equals this keras_key. Slower, only used if the fast path misses.
        for torch_key in self.state_dict:
            if self.name_map(torch_key) == keras_key:
                return torch_key
        return None


def load_torch_state_dict_as_numpy(checkpoint_path, key_prefix_strip=None):
    """Loads a PyTorch `.pt`/`.pth`/`.bin` checkpoint and returns a
    dict[str, np.ndarray]. Requires `torch` to be installed; only used at
    conversion time, never at inference/training time.

    `key_prefix_strip`: an optional string prefix to strip from every key
    (e.g. "module." from a DDP-saved checkpoint, or "model." from a
    Lightning checkpoint).
    """
    import torch  # local import: torch is a conversion-time-only dependency

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:  # common Lightning/HF wrapper
        state_dict = state_dict["state_dict"]
    if "model" in state_dict and all(isinstance(v, dict) for v in [state_dict.get("model", {})]):
        state_dict = state_dict.get("model", state_dict)

    out = {}
    for k, v in state_dict.items():
        if key_prefix_strip and k.startswith(key_prefix_strip):
            k = k[len(key_prefix_strip):]
        out[k] = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    return out


def load_safetensors_as_numpy(checkpoint_path):
    """Loads a `.safetensors` checkpoint (common for HF-hosted foundation
    models like Prithvi/Clay/CROMA) directly to numpy, no torch required."""
    from safetensors import safe_open

    out = {}
    with safe_open(checkpoint_path, framework="numpy") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out
