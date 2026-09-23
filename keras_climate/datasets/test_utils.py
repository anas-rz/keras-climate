import hashlib
import os

import numpy as np
import pytest
from keras import ops

from keras_climate.datasets.utils import (
    BoundingBox,
    array_to_tensor,
    check_integrity,
    concat_samples,
    disambiguate_timestamp,
    extract_archive,
    find_files,
    merge_samples,
    quantile_normalization,
    rgb_to_mask,
    stack_samples,
    unbind_samples,
)


class TestBoundingBox:
    def test_valid(self):
        bbox = BoundingBox(0, 1, 0, 1, 0, 1)
        assert bbox.minx == 0

    def test_invalid_x(self):
        with pytest.raises(ValueError, match="Bounding box is invalid"):
            BoundingBox(1, 0, 0, 1, 0, 1)

    def test_getitem(self):
        bbox = BoundingBox(0, 1, 2, 3, 4, 5)
        assert list(bbox) == [0, 1, 2, 3, 4, 5]

    def test_contains(self):
        outer = BoundingBox(0, 10, 0, 10, 0, 10)
        inner = BoundingBox(1, 2, 1, 2, 1, 2)
        assert inner in outer
        assert outer not in inner

    def test_or(self):
        a = BoundingBox(0, 1, 0, 1, 0, 1)
        b = BoundingBox(1, 2, 1, 2, 1, 2)
        result = a | b
        assert result == BoundingBox(0, 2, 0, 2, 0, 2)

    def test_and_overlap(self):
        a = BoundingBox(0, 2, 0, 2, 0, 2)
        b = BoundingBox(1, 3, 1, 3, 1, 3)
        result = a & b
        assert result == BoundingBox(1, 2, 1, 2, 1, 2)

    def test_and_no_overlap_raises(self):
        a = BoundingBox(0, 1, 0, 1, 0, 1)
        b = BoundingBox(5, 6, 5, 6, 5, 6)
        with pytest.raises(ValueError, match="do not overlap"):
            a & b

    def test_area_and_volume(self):
        bbox = BoundingBox(0, 2, 0, 3, 0, 1)
        assert bbox.area == 6
        assert bbox.volume == 6

    def test_intersects(self):
        a = BoundingBox(0, 2, 0, 2, 0, 2)
        b = BoundingBox(1, 3, 1, 3, 1, 3)
        c = BoundingBox(10, 11, 10, 11, 10, 11)
        assert a.intersects(b)
        assert not a.intersects(c)

    def test_split_horizontal(self):
        bbox = BoundingBox(0, 10, 0, 10, 0, 1)
        left, right = bbox.split(0.5)
        assert left.maxx == 5
        assert right.minx == 5

    def test_split_invalid_proportion(self):
        bbox = BoundingBox(0, 10, 0, 10, 0, 1)
        with pytest.raises(ValueError, match="proportion must be between"):
            bbox.split(1.5)


def test_check_integrity_missing_file(tmp_path):
    assert not check_integrity(str(tmp_path / "nope.txt"))


def test_check_integrity_no_checksum(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("hello")
    assert check_integrity(str(f))


def test_check_integrity_matching_md5(tmp_path):
    f = tmp_path / "f.txt"
    f.write_bytes(b"hello world")
    md5 = hashlib.md5(b"hello world").hexdigest()
    assert check_integrity(str(f), md5=md5)


def test_check_integrity_mismatched_md5(tmp_path):
    f = tmp_path / "f.txt"
    f.write_bytes(b"hello world")
    assert not check_integrity(str(f), md5="0" * 32)


def test_extract_zip(tmp_path):
    import zipfile

    src = tmp_path / "a.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("inner.txt", "contents")

    out_dir = tmp_path / "out"
    extract_archive(str(src), str(out_dir))
    assert (out_dir / "inner.txt").read_text() == "contents"


def test_disambiguate_timestamp_year():
    mint, maxt = disambiguate_timestamp("2021", "%Y")
    assert mint.year == 2021
    assert maxt.year == 2021
    assert maxt.month == 12


def test_disambiguate_timestamp_no_temporal_info():
    import pandas as pd

    mint, maxt = disambiguate_timestamp("", "")
    assert mint == pd.Timestamp.min
    assert maxt == pd.Timestamp.max


def test_stack_samples():
    samples = [{"image": np.ones((2, 2))}, {"image": np.zeros((2, 2))}]
    out = stack_samples(samples)
    assert tuple(out["image"].shape) == (2, 2, 2)


def test_concat_samples():
    # Channel-last (H, W, C) images: concatenation happens along the last
    # (channel) axis, not axis 0.
    samples = [
        {"image": np.ones((4, 4, 2))},
        {"image": np.zeros((4, 4, 3))},
    ]
    out = concat_samples(samples)
    assert tuple(out["image"].shape) == (4, 4, 5)


def test_concat_samples_1d_metadata_unaffected():
    # 1-D metadata (e.g. "bounds") has only one axis, so axis=-1 concat
    # matches the previous axis=0 behavior.
    samples = [{"bounds": np.arange(9)}, {"bounds": np.arange(9)}]
    out = concat_samples(samples)
    assert tuple(out["bounds"].shape) == (18,)


def test_merge_samples_takes_max():
    samples = [{"image": np.array([0.0, 5.0])}, {"image": np.array([3.0, 1.0])}]
    out = merge_samples(samples)
    np.testing.assert_array_equal(ops.convert_to_numpy(out["image"]), [3.0, 5.0])


def test_unbind_samples_reverses_stack():
    samples = [{"image": np.ones((2, 2)) * i} for i in range(3)]
    stacked = stack_samples(samples)
    unbound = unbind_samples(stacked)
    assert len(unbound) == 3
    for i, sample in enumerate(unbound):
        np.testing.assert_array_equal(
            ops.convert_to_numpy(sample["image"]), np.ones((2, 2)) * i
        )


def test_array_to_tensor_uint16():
    arr = np.array([1, 2, 3], dtype=np.uint16)
    tensor = array_to_tensor(arr)
    assert ops.convert_to_numpy(tensor).dtype == np.int32


def test_array_to_tensor_uint32():
    import keras

    arr = np.array([1, 2, 3], dtype=np.uint32)
    tensor = array_to_tensor(arr)
    dtype = ops.convert_to_numpy(tensor).dtype
    if keras.backend.backend() == "jax":
        # JAX disables 64-bit precision by default (a well-known JAX
        # limitation, not something array_to_tensor controls), so an int64
        # request is silently downcast to int32 unless the user opts in via
        # `jax.config.update("jax_enable_x64", True)`.
        assert dtype in (np.int32, np.int64)
    else:
        assert dtype == np.int64


def test_quantile_normalization_clips_to_unit_range():
    img = np.linspace(0, 100, 1000).astype("float32")
    out = ops.convert_to_numpy(quantile_normalization(img, lower=0.02, upper=0.98))
    assert out.min() >= 0
    assert out.max() <= 1


def test_quantile_normalization_all_nodata_returns_input():
    img = np.zeros((4, 4), dtype="float32")
    out = quantile_normalization(img, nodata=0)
    np.testing.assert_array_equal(ops.convert_to_numpy(ops.convert_to_tensor(out)), img)


def test_rgb_to_mask():
    rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0)]
    mask = rgb_to_mask(rgb, colors)
    np.testing.assert_array_equal(mask, [[0, 1]])


def test_find_files_dir(tmp_path):
    (tmp_path / "a.tif").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    files = find_files(str(tmp_path), "*.tif")
    assert len(files) == 1
    assert files[0].endswith("a.tif")


def test_find_files_single_file(tmp_path):
    f = tmp_path / "a.tif"
    f.write_text("x")
    files = find_files(str(f), "*.tif")
    assert files == [str(f)]
