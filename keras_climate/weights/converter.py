import re
import numpy as np


def infer_transpose(torch_shape, keras_shape, param_kind=None):
    if param_kind == "dense_kernel":
        return (1, 0)

    if tuple(torch_shape) == tuple(keras_shape):
        return None

    if len(torch_shape) == 2 and torch_shape[::-1] == tuple(keras_shape):
        return (1, 0)

    if param_kind == "conv2d_kernel" or len(torch_shape) == 4:
        candidate = (torch_shape[2], torch_shape[3], torch_shape[1], torch_shape[0])
        if candidate == tuple(keras_shape):
            return (2, 3, 1, 0)

    if param_kind == "conv3d_kernel" or len(torch_shape) == 5:
        candidate = (torch_shape[2], torch_shape[3], torch_shape[4], torch_shape[1], torch_shape[0])
        if candidate == tuple(keras_shape):
            return (2, 3, 4, 1, 0)

    if param_kind == "conv1d_kernel" or len(torch_shape) == 3:
        candidate = (torch_shape[2], torch_shape[1], torch_shape[0])
        if candidate == tuple(keras_shape):
            return (2, 1, 0)

    if param_kind == "depthwise_kernel" and len(torch_shape) == 4:
        return "depthwise"

    return None


def apply_transpose(array, perm):
    if perm is None:
        return array
    if perm == "depthwise":
        out_ch, _, kh, kw = array.shape
        arr = array.reshape(out_ch, 1, kh, kw)
        arr = np.transpose(arr, (2, 3, 0, 1))
        return arr
    return np.transpose(array, perm)


def build_regex_mapper(rules):

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, replacement in compiled:
            if pattern.search(torch_key):
                return pattern.sub(replacement, torch_key)
        return None

    return mapper


class WeightConverter:
    def __init__(self, keras_model, torch_state_dict, name_map, param_kind_map=None,
                 skip_patterns=None):
        self.model = keras_model
        self.state_dict = torch_state_dict
        self.name_map = name_map if callable(name_map) else build_regex_mapper(name_map)
        self.param_kind_map = param_kind_map or {}
        self.skip_patterns = [re.compile(p) for p in (skip_patterns or [])]

    def convert(self, strict=False, verbose=True):
        keras_by_name = {w.path: w for w in self.model.weights}
        if not keras_by_name:
            keras_by_name = {w.name: w for w in self.model.weights}

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
        direct = self.name_map(keras_key)
        if direct is not None and direct in self.state_dict:
            return direct

        for torch_key in self.state_dict:
            if self.name_map(torch_key) == keras_key:
                return torch_key
        return None


def load_torch_state_dict_as_numpy(checkpoint_path, key_prefix_strip=None):
    import torch

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
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
    from safetensors import safe_open

    out = {}
    with safe_open(checkpoint_path, framework="numpy") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out
