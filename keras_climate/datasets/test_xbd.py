import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.xbd import XView2, xBD, xBDDistShift


def _write_image(path, size=8):
    array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path, size=8):
    array = np.random.randint(0, 5, (size, size), dtype=np.uint8)
    Image.fromarray(array, mode="L").save(path)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    for split_dir in ("train", "test"):
        image_dir = os.path.join(root, split_dir, "images")
        mask_dir = os.path.join(root, split_dir, "targets")
        os.makedirs(image_dir)
        os.makedirs(mask_dir)
        for disaster, idx in (("hurricane-harvey", "00000072"), ("hurricane-michael", "00000105")):
            name = f"{disaster}_{idx}"
            for suffix in ("pre_disaster", "post_disaster"):
                _write_image(os.path.join(image_dir, f"{name}_{suffix}.png"))
                _write_mask(os.path.join(mask_dir, f"{name}_{suffix}_target.png"))
    return root


def test_getitem(prepared_root):
    ds = xBD(root=prepared_root, split="train")
    sample = ds[0]
    assert tuple(sample["image"].shape)[0] == 2
    assert sample["image"].ndim == 4
    assert sample["mask"].ndim == 2


def test_len(prepared_root):
    ds = xBD(root=prepared_root, split="train")
    assert len(ds) == 2


def test_deprecated_alias(prepared_root):
    ds = XView2(root=prepared_root, split="train")
    assert len(ds) == 2


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        xBD(root=prepared_root, split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        xBD(root=str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = xBD(root=prepared_root, split="train")
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()
    ds.plot(sample, show_titles=False)
    plt.close()
    sample["prediction"] = sample["mask"]
    ds.plot(sample)
    plt.close()


def test_dist_shift(prepared_root):
    ds = xBDDistShift(
        root=prepared_root,
        split="train",
        id_disaster="hurricane-harvey",
        ood_disaster="hurricane-michael",
    )
    sample = ds[0]
    assert sample["image"].ndim == 3
    unique_vals = set(np.unique(np.asarray(sample["mask"])).tolist())
    assert unique_vals <= {0, 1}
    assert "hurricane-harvey" in ds.files[0]["image"]


def test_dist_shift_pre_post_both(prepared_root):
    ds = xBDDistShift(
        root=prepared_root,
        split="train",
        id_disaster="hurricane-harvey",
        id_pre_post="both",
        ood_disaster="hurricane-michael",
    )
    assert len(ds) == 4


def test_dist_shift_invalid_config(prepared_root):
    with pytest.raises(AssertionError):
        xBDDistShift(
            root=prepared_root,
            id_disaster="hurricane-harvey",
            ood_disaster="hurricane-harvey",
        )


def test_dist_shift_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = xBDDistShift(
        root=prepared_root,
        split="train",
        id_disaster="hurricane-harvey",
        ood_disaster="hurricane-michael",
    )
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()
    sample["prediction"] = sample["mask"]
    ds.plot(sample)
    plt.close()
