"""Shared test-only helpers for keras_climate.datasets tests (not a test module itself)."""


def assert_dtype(tensor, expected, jax_fallback=None):
    """Assert a tensor's dtype equals *expected*, portable across backends.

    Keras backends disagree on whether the raw ``tensor.dtype`` object
    supports string equality: it works for TensorFlow and JAX, but
    ``torch.dtype.__eq__`` does not accept a plain string (``torch.float32
    == "float32"`` is ``False``). ``keras.backend.standardize_dtype``
    normalizes it into a plain, comparable string on every backend.

    Args:
        tensor: a Keras tensor.
        expected: the expected dtype string (e.g. ``"float32"``).
        jax_fallback: an additional dtype string to also accept, but only
            on the jax backend. Use this for int64 assertions, since JAX
            disables 64-bit precision by default (a well-known JAX
            limitation, not something keras_climate's datasets control),
            silently downcasting int64 requests to int32.
    """
    import keras

    actual = keras.backend.standardize_dtype(tensor.dtype)
    if jax_fallback is not None and keras.backend.backend() == "jax":
        assert actual in (expected, jax_fallback), actual
    else:
        assert actual == expected, actual


def assert_int64_dtype(tensor):
    """Assert an integer tensor is int64, tolerating JAX's 32-bit default."""
    assert_dtype(tensor, "int64", jax_fallback="int32")
