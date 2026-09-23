import numpy as np
import pytest
from keras import ops

from keras_climate.transforms import SatSlideMix


def test_gamma_multiplies_batch():
    x = np.random.rand(2, 8, 8, 3).astype("float32")
    aug = SatSlideMix(gamma=3, p=1.0)
    out = aug(x)
    assert tuple(out.shape) == (6, 8, 8, 3)


def test_preserves_pixel_values_via_permutation():
    """Rolling doesn't change the set of pixel values, only their arrangement."""
    x = np.arange(8 * 8 * 3, dtype="float32").reshape(1, 8, 8, 3)
    aug = SatSlideMix(gamma=1, p=1.0)
    out = ops.convert_to_numpy(aug(x))
    np.testing.assert_allclose(np.sort(out.ravel()), np.sort(x.ravel()))


def test_p_zero_is_identity_up_to_repeat():
    x = np.random.rand(2, 8, 8, 3).astype("float32")
    aug = SatSlideMix(gamma=2, p=0.0)
    out = ops.convert_to_numpy(aug(x))
    expected = np.repeat(x, 2, axis=0)
    np.testing.assert_array_equal(out, expected)


def test_unbatched_input():
    x = np.random.rand(8, 8, 3).astype("float32")
    aug = SatSlideMix(gamma=2, p=1.0)
    out = aug(x)
    assert tuple(out.shape) == (2, 8, 8, 3)


def test_not_training_returns_repeated_input():
    x = np.random.rand(2, 8, 8, 3).astype("float32")
    aug = SatSlideMix(gamma=2, p=1.0)
    out = ops.convert_to_numpy(aug(x, training=False))
    np.testing.assert_array_equal(out, np.repeat(x, 2, axis=0))


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_invalid_gamma(bad):
    with pytest.raises(AssertionError, match="gamma"):
        SatSlideMix(gamma=bad)
