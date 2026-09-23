import numpy as np
from keras import ops

from keras_climate.transforms import Rearrange


def test_insert_time_dimension():
    x = np.random.rand(2, 3, 8, 8).astype("float32")
    out = Rearrange("b (t c) h w -> b t c h w", c=1)(x)
    assert tuple(out.shape) == (2, 3, 1, 8, 8)


def test_collapse_time_dimension():
    x = np.random.rand(2, 3, 1, 8, 8).astype("float32")
    out = Rearrange("b t c h w -> b (t c) h w")(x)
    assert tuple(out.shape) == (2, 3, 8, 8)


def test_channel_permute_roundtrip():
    x = np.arange(2 * 4 * 4 * 3, dtype="float32").reshape(2, 4, 4, 3)
    to_chw = Rearrange("b h w c -> b c h w")
    to_hwc = Rearrange("b c h w -> b h w c")
    out = ops.convert_to_numpy(to_hwc(to_chw(x)))
    np.testing.assert_array_equal(out, x)
