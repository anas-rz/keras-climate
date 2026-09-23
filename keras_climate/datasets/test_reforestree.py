import csv
import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.reforestree import ReforesTree


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    image_paths = [
        os.path.join("tiles", "Site1", "Site1_RGB_0_0_0_4000_4000.png"),
        os.path.join("tiles", "Site2", "Site2_RGB_0_0_0_4000_4000.png"),
    ]

    for rel_path in image_paths:
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = (np.random.rand(16, 16, 3) * 255).astype("uint8")
        Image.fromarray(data).save(path)

    mapping_dir = os.path.join(root, "mapping")
    os.makedirs(mapping_dir, exist_ok=True)
    with open(os.path.join(mapping_dir, "final_dataset.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["img_path", "xmin", "ymin", "xmax", "ymax", "group", "AGB"])
        for rel_path in image_paths:
            fname = os.path.basename(rel_path)
            writer.writerow([fname, 0, 0, 8, 8, "banana", 6.75])
            writer.writerow([fname, 8, 8, 16, 16, "cacao", 6.75])

    return root


def test_getitem(prepared_root):
    ds = ReforesTree(root=prepared_root, download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert sample["bbox_xyxy"].shape[-1] == 4
    assert "label" in sample
    assert "agb" in sample


def test_len(prepared_root):
    ds = ReforesTree(root=prepared_root, download=False)
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        ReforesTree(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ReforesTree(root=prepared_root, download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
