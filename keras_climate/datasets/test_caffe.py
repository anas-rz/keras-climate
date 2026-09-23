import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.caffe import CaFFe

IMG_SIZE = 8


def _write(path, values, size=(IMG_SIZE, IMG_SIZE)):
    data = np.random.choice(values, size=size).astype("uint8")
    Image.fromarray(data).save(path)


@pytest.fixture(params=["train", "val", "test"])
def prepared_root(tmp_path, request):
    split = request.param
    root_dir = os.path.join(str(tmp_path), "caffe")
    for sub in ("zones", "sar_images", "fronts"):
        os.makedirs(os.path.join(root_dir, sub, split), exist_ok=True)

    filenames = [
        "Crane_2002-11-09_ERS_20_2_061_zones__93_102_0_0_0.png",
        "JAC_2015-12-23_TSX_6_1_005_zones__57_49_195_384_1024.png",
    ]
    for fname in filenames:
        _write(os.path.join(root_dir, "zones", split, fname), [0, 64, 127, 254])
        _write(
            os.path.join(root_dir, "sar_images", split, fname.replace("_zones_", "_")),
            list(range(256)),
        )
        _write(
            os.path.join(root_dir, "fronts", split, fname.replace("_zones_", "_front_")),
            [0, 255],
        )

    return str(tmp_path), split


def test_getitem(prepared_root):
    root, split = prepared_root
    ds = CaFFe(root=root, split=split)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (IMG_SIZE, IMG_SIZE, 1)
    assert sample["mask_front"].shape == (IMG_SIZE, IMG_SIZE)
    assert sample["mask_zones"].shape == (IMG_SIZE, IMG_SIZE)


def test_len(prepared_root):
    root, split = prepared_root
    ds = CaFFe(root=root, split=split)
    assert len(ds) == 2


def test_invalid_split(tmp_path):
    with pytest.raises(AssertionError):
        CaFFe(root=str(tmp_path), split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        CaFFe(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, split = prepared_root
    ds = CaFFe(root=root, split=split)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()
    sample["prediction"] = sample["mask_front"]
    ds.plot(sample)
    plt.close()
