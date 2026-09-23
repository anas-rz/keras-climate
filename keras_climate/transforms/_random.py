"""Internal helper for probability-gated (per-sample) random transforms."""

import keras
from keras import ops


def bernoulli_mask(batch_size, p, same_on_batch=False):
    """A (batch_size,) float mask of 1.0 (apply) / 0.0 (skip), drawn with prob *p*."""
    n = 1 if same_on_batch else batch_size
    u = keras.random.uniform((n,))
    mask = ops.cast(u < p, "float32")
    if same_on_batch:
        mask = ops.repeat(mask, batch_size)
    return mask


def ensure_batched(x):
    """Add a leading batch dim if *x* is a single (unbatched) sample.

    Returns the (possibly expanded) tensor and whether it was expanded.
    """
    x = ops.convert_to_tensor(x)
    if len(x.shape) == 3:
        return ops.expand_dims(x, 0), True
    return x, False
