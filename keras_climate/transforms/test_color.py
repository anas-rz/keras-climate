import numpy as np
import pytest
from keras import ops

from keras_climate.transforms import RandomGrayscale


@pytest.fixture
def sample():
    return np.arange(4 * 4 * 3, dtype="float32").reshape(4, 4, 3)


@pytest.fixture
def batch():
    return np.arange(2 * 4 * 4 * 3, dtype="float32").reshape(2, 4, 4, 3)


@pytest.mark.parametrize(
    "weights", [[1.0, 1.0, 1.0], [0.299, 0.587, 0.114], [1.0, 2.0, 3.0]]
)
def test_random_grayscale_sample(weights, sample):
    aug = RandomGrayscale(weights, p=1)
    output = ops.convert_to_numpy(aug(sample))
    assert output.shape == sample.shape
    # All channels should be identical (grayscale)
    np.testing.assert_allclose(output[..., 0], output[..., 1], atol=1e-4)
    np.testing.assert_allclose(output[..., 0], output[..., 2], atol=1e-4)


@pytest.mark.parametrize(
    "weights", [[1.0, 1.0, 1.0], [0.299, 0.587, 0.114], [1.0, 2.0, 3.0]]
)
def test_random_grayscale_batch(weights, batch):
    aug = RandomGrayscale(weights, p=1)
    output = ops.convert_to_numpy(aug(batch))
    assert output.shape == batch.shape
    np.testing.assert_allclose(output[..., 0], output[..., 1], atol=1e-4)


def test_random_grayscale_matches_formula(sample):
    weights = np.array([0.299, 0.587, 0.114], dtype="float32")
    weights = weights / weights.sum()
    aug = RandomGrayscale(weights, p=1)
    output = ops.convert_to_numpy(aug(sample))
    expected = (sample * weights).sum(axis=-1, keepdims=True)
    expected = np.repeat(expected, 3, axis=-1)
    np.testing.assert_allclose(output, expected, atol=1e-4)


def test_p_zero_is_identity(sample):
    aug = RandomGrayscale([1.0, 1.0, 1.0], p=0.0)
    output = ops.convert_to_numpy(aug(sample))
    np.testing.assert_array_equal(output, sample)
