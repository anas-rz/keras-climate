import numpy as np
import pytest
from keras import ops

from keras_climate.transforms import (
    AppendBNDVI,
    AppendEVI,
    AppendGBNDVI,
    AppendGNDVI,
    AppendGRNDVI,
    AppendMNDWI,
    AppendNBR,
    AppendNDBI,
    AppendNDRE,
    AppendNDSI,
    AppendNDVI,
    AppendNDWI,
    AppendNormalizedDifferenceIndex,
    AppendRBNDVI,
    AppendSAVI,
    AppendSWI,
    AppendTriBandNormalizedDifferenceIndex,
)


@pytest.fixture
def sample():
    return np.arange(4 * 4 * 3, dtype="float32").reshape(4, 4, 3)


@pytest.fixture
def batch():
    return np.arange(2 * 4 * 4 * 3, dtype="float32").reshape(2, 4, 4, 3)


def test_append_index_sample(sample):
    h, w, c = sample.shape
    out = AppendNormalizedDifferenceIndex(index_a=0, index_b=1)(sample)
    assert tuple(out.shape) == (h, w, c + 1)


def test_append_index_batch(batch):
    b, h, w, c = batch.shape
    out = AppendNormalizedDifferenceIndex(index_a=0, index_b=1)(batch)
    assert tuple(out.shape) == (b, h, w, c + 1)


def test_append_triband_index_batch(batch):
    b, h, w, c = batch.shape
    out = AppendTriBandNormalizedDifferenceIndex(index_a=0, index_b=1, index_c=2)(batch)
    assert tuple(out.shape) == (b, h, w, c + 1)


@pytest.mark.parametrize(
    "index_cls",
    [
        AppendBNDVI,
        AppendNBR,
        AppendNDBI,
        AppendNDRE,
        AppendNDSI,
        AppendNDVI,
        AppendNDWI,
        AppendMNDWI,
        AppendSWI,
        AppendGNDVI,
    ],
)
def test_append_normalized_difference_indices(sample, index_cls):
    h, w, c = sample.shape
    out = index_cls(0, 1)(sample)
    assert tuple(out.shape) == (h, w, c + 1)


@pytest.mark.parametrize("index_cls", [AppendGBNDVI, AppendGRNDVI, AppendRBNDVI])
def test_append_tri_band_normalized_difference_indices(sample, index_cls):
    h, w, c = sample.shape
    out = index_cls(0, 1, 2)(sample)
    assert tuple(out.shape) == (h, w, c + 1)


def test_append_savi(sample):
    h, w, c = sample.shape
    out = AppendSAVI(index_nir=0, index_red=1)(sample)
    assert tuple(out.shape) == (h, w, c + 1)


def test_append_evi(sample):
    h, w, c = sample.shape
    out = AppendEVI(index_nir=0, index_red=1, index_blue=2)(sample)
    assert tuple(out.shape) == (h, w, c + 1)


def test_ndvi_formula():
    x = np.zeros((1, 1, 2), dtype="float32")
    x[..., 0] = 0.8  # nir
    x[..., 1] = 0.2  # red
    out = ops.convert_to_numpy(AppendNDVI(index_nir=0, index_red=1)(x))
    expected = (0.8 - 0.2) / (0.8 + 0.2)
    np.testing.assert_allclose(out[..., -1], expected, atol=1e-4)


def test_grndvi_formula():
    x = np.zeros((1, 1, 3), dtype="float32")
    x[..., 0] = 0.8  # nir
    x[..., 1] = 0.1  # green
    x[..., 2] = 0.2  # red
    out = ops.convert_to_numpy(
        AppendGRNDVI(index_nir=0, index_green=1, index_red=2)(x)
    )
    expected = (0.8 - (0.1 + 0.2)) / (0.8 + (0.1 + 0.2))
    np.testing.assert_allclose(out[..., -1], expected, atol=1e-4)


def test_savi_formula():
    x = np.zeros((1, 1, 2), dtype="float32")
    x[..., 0] = 0.8  # nir
    x[..., 1] = 0.2  # red
    out = ops.convert_to_numpy(AppendSAVI(index_nir=0, index_red=1)(x))
    expected = 1.5 * (0.8 - 0.2) / (0.8 + 0.2 + 0.5)
    np.testing.assert_allclose(out[..., -1], expected, atol=1e-4)


def test_evi_formula():
    x = np.zeros((1, 1, 3), dtype="float32")
    x[..., 0] = 0.8  # nir
    x[..., 1] = 0.2  # red
    x[..., 2] = 0.1  # blue
    out = ops.convert_to_numpy(
        AppendEVI(index_nir=0, index_red=1, index_blue=2)(x)
    )
    expected = 2.5 * (0.8 - 0.2) / (0.8 + 6 * 0.2 - 7.5 * 0.1 + 1)
    np.testing.assert_allclose(out[..., -1], expected, atol=1e-4)
