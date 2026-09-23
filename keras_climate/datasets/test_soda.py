import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.soda import SODAA


def _write_annotation(path, polys, category_ids):
    data = {
        "annotations": [
            {"poly": poly, "category_id": cid}
            for poly, cid in zip(polys, category_ids)
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    images_dir = os.path.join(root, "Images")
    labels_dir = os.path.join(root, "Annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    rows = []
    for i in range(2):
        img_name = f"img{i}.jpg"
        label_name = f"img{i}.json"
        img = Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype="uint8"))
        img.save(os.path.join(images_dir, img_name))
        _write_annotation(
            os.path.join(labels_dir, label_name),
            polys=[[1, 1, 5, 1, 5, 5, 1, 5]],
            category_ids=[0],
        )
        rows.append(
            {
                "split": "train",
                "image_path": os.path.join("Images", img_name),
                "label_path": os.path.join("Annotations", label_name),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(root, "sample_df.csv"), index=False)
    return root


def test_getitem_horizontal(prepared_root):
    ds = SODAA(root=prepared_root, split="train", bbox_orientation="horizontal")
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert "bbox_xyxy" in sample
    assert tuple(sample["bbox_xyxy"].shape) == (1, 4)
    assert "label" in sample


def test_getitem_oriented(prepared_root):
    ds = SODAA(root=prepared_root, split="train", bbox_orientation="oriented")
    sample = ds[0]
    assert "bbox" in sample
    assert tuple(sample["bbox"].shape) == (1, 8)


def test_len(prepared_root):
    ds = SODAA(root=prepared_root, split="train")
    assert len(ds) == 2


def test_invalid_split_raises():
    with pytest.raises(AssertionError):
        SODAA(split="bogus")


def test_invalid_orientation_raises():
    with pytest.raises(AssertionError):
        SODAA(bbox_orientation="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        SODAA(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = SODAA(root=prepared_root, split="train", bbox_orientation="horizontal")
    ds.plot(ds[0], suptitle="Test")
    plt.close()
