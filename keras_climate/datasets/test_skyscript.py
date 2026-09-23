import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("tokenizers")

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.skyscript import SkyScript


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    for directory in SkyScript.image_dirs:
        os.makedirs(os.path.join(root, directory), exist_ok=True)

    img_rel_path = os.path.join(SkyScript.image_dirs[0], "img0.jpg")
    img = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype="uint8"))
    img.save(os.path.join(root, img_rel_path))

    captions = pd.DataFrame(
        {"filepath": [img_rel_path], "title": ["a remote sensing image of a field"]}
    )
    for split, fname in SkyScript.caption_files.items():
        captions.to_csv(os.path.join(root, fname), index=False)

    return root


def test_getitem(prepared_root):
    ds = SkyScript(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)
    assert "caption" in sample


def test_len(prepared_root):
    ds = SkyScript(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_invalid_split_raises():
    with pytest.raises(AssertionError):
        SkyScript(split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SkyScript(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SkyScript(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
