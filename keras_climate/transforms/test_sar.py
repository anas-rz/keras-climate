import numpy as np
import pytest
from keras import ops

from keras_climate.transforms import LeeFilter
from keras_climate.transforms.sar import lee_filter

scipy = pytest.importorskip("scipy", minversion="1.11.2")
from scipy.ndimage import uniform_filter  # noqa: E402


def _numpy_lee_reference(image, window_size, num_looks, eps=1e-8):
    """Reference Lee filter using scipy.ndimage on a 2D NumPy array."""
    image = image.astype(np.float64)
    sigma_v_sq = 1.0 / num_looks
    mean_local = uniform_filter(image, size=window_size, mode="mirror")
    mean_sq_local = uniform_filter(image * image, size=window_size, mode="mirror")
    var_local = np.clip(mean_sq_local - mean_local**2, 0.0, None)
    var_signal = np.clip(var_local - sigma_v_sq * mean_local**2, 0.0, None)
    weight = var_signal / (var_signal + sigma_v_sq * mean_local**2 + eps)
    return mean_local + weight * (image - mean_local)


def _make_synthetic_sar(seed=0, size=64):
    """Bright square on dark background with unit-mean exponential speckle."""
    rng = np.random.default_rng(seed)
    signal = np.ones((size, size), dtype=np.float64)
    signal[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 5.0
    speckle = rng.exponential(scale=1.0, size=(size, size))
    return signal * speckle


@pytest.mark.parametrize("window_size", [3, 5, 7])
@pytest.mark.parametrize("num_looks", [1.0, 5.0])
def test_matches_numpy_reference(window_size, num_looks):
    img_np = _make_synthetic_sar()
    ref = _numpy_lee_reference(img_np, window_size, num_looks)
    # (H, W) -> (1, H, W, 1) channels-last
    x = img_np[None, :, :, None].astype("float32")
    out = ops.convert_to_numpy(lee_filter(x, window_size=window_size, num_looks=num_looks))
    out_np = out[0, :, :, 0]
    np.testing.assert_allclose(out_np, ref, rtol=1e-4, atol=1e-4)


def test_preserves_shape():
    x = np.random.rand(2, 16, 16, 3).astype("float32")
    out = lee_filter(x, window_size=5)
    assert tuple(out.shape) == x.shape


def test_non_negative_output():
    x = (np.random.rand(1, 16, 16, 1) * 10).astype("float32")
    out = ops.convert_to_numpy(lee_filter(x, window_size=7))
    assert (out >= 0).all()


def test_reduces_variance_in_homogeneous_region():
    rng = np.random.default_rng(42)
    speckle = (rng.exponential(scale=1.0, size=(64, 64)) * 3.0).astype("float32")
    x = speckle[None, :, :, None]
    out = ops.convert_to_numpy(lee_filter(x, window_size=9))[0, :, :, 0]
    assert out.var() < speckle.var() * 0.5


@pytest.mark.parametrize("bad", [0, -1, 4, 8])
def test_rejects_invalid_window_size(bad):
    with pytest.raises(ValueError, match="window_size"):
        lee_filter(np.zeros((1, 8, 8, 1), dtype="float32"), window_size=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_invalid_num_looks(bad):
    with pytest.raises(ValueError, match="num_looks"):
        lee_filter(np.zeros((1, 8, 8, 1), dtype="float32"), window_size=5, num_looks=bad)


def test_lee_filter_class_sample():
    x = np.random.rand(8, 8, 1).astype("float32")
    aug = LeeFilter(window_size=5, p=1.0)
    output = aug(x)
    assert tuple(output.shape) == x.shape


def test_lee_filter_class_batch():
    x = np.random.rand(2, 8, 8, 1).astype("float32")
    aug = LeeFilter(window_size=5, p=1.0)
    output = aug(x)
    assert tuple(output.shape) == x.shape


def test_p_zero_is_identity():
    x = np.random.rand(2, 8, 8, 1).astype("float32")
    aug = LeeFilter(p=0.0)
    output = ops.convert_to_numpy(aug(x))
    np.testing.assert_array_equal(output, x)


@pytest.mark.parametrize("bad", [0, -1, 4, 8])
def test_class_rejects_invalid_window_size(bad):
    with pytest.raises(ValueError, match="window_size"):
        LeeFilter(window_size=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_class_rejects_invalid_num_looks(bad):
    with pytest.raises(ValueError, match="num_looks"):
        LeeFilter(num_looks=bad)
