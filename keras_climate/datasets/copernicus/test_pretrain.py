import numpy as np
import pytest

from keras_climate.datasets.copernicus.pretrain import CopernicusPretrain
from keras_climate.datasets.errors import DependencyNotFoundError


@pytest.fixture
def dataset():
    # Bypass __init__ (which requires the optional `webdataset` dependency)
    # to unit test the pure sample-transformation helpers directly.
    return object.__new__(CopernicusPretrain)


def test_missing_webdataset_dependency(tmp_path):
    with pytest.raises(DependencyNotFoundError):
        CopernicusPretrain(urls=str(tmp_path / "example-{000000..000000}.tar"))


def test_drop_metadata(dataset):
    sample = {
        "s1_grd.pth": np.zeros((2, 4, 4)),
        "__key__": "example-000000",
        "__url__": "data/example-000000.tar",
    }
    out = dataset._drop_metadata(sample)
    assert set(out) == {"s1_grd.pth"}


def test_has_all_modalities(dataset):
    required_keys = [
        "s1_grd.pth",
        "s2_toa.pth",
        "s3_olci.pth",
        "s5p_co.pth",
        "s5p_no2.pth",
        "s5p_o3.pth",
        "s5p_so2.pth",
        "dem.pth",
    ]
    full_sample = {key: np.zeros((1, 2, 2)) for key in required_keys}
    assert dataset._has_all_modalities(full_sample) is True

    partial_sample = {key: np.zeros((1, 2, 2)) for key in required_keys[:-1]}
    assert dataset._has_all_modalities(partial_sample) is False


def test_sample_one_local_patch(dataset):
    sample = {
        "s1_grd.pth": np.random.rand(5, 2, 4, 4),
        "s2_toa.pth": np.random.rand(5, 3, 4, 4),
    }
    out = dataset._sample_one_local_patch(sample)
    assert out["s1_grd.pth"].shape == (2, 4, 4)
    assert out["s2_toa.pth"].shape == (3, 4, 4)


def test_sample_one_time_stamp(dataset):
    sample = {
        "s1_grd.pth": np.random.rand(3, 2, 4, 4),
        "dem.pth": np.random.rand(4, 4),
    }
    out = dataset._sample_one_time_stamp(sample)
    assert out["s1_grd.pth"].shape == (2, 4, 4)
    # dem.pth has no time dimension and should be left untouched
    assert out["dem.pth"].shape == (4, 4)


def test_to_tensors(dataset):
    sample = {"s1_grd.pth": np.random.rand(2, 4, 4)}
    out = dataset._to_tensors(sample)
    assert tuple(out["s1_grd.pth"].shape) == (2, 4, 4)
