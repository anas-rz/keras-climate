import os
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.dior import DIOR, parse_pascal_voc


def _write_image(path, size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.randint(0, 255, (size, size, 3), dtype="uint8")
    Image.fromarray(arr).save(path)


def _write_annotation(path, image_name, classes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    root = ElementTree.Element("annotation")
    ElementTree.SubElement(root, "filename").text = image_name
    obj = ElementTree.SubElement(root, "object")
    ElementTree.SubElement(obj, "name").text = classes[0]
    bbox = ElementTree.SubElement(obj, "bndbox")
    ElementTree.SubElement(bbox, "xmin").text = "1"
    ElementTree.SubElement(bbox, "ymin").text = "1"
    ElementTree.SubElement(bbox, "xmax").text = "5"
    ElementTree.SubElement(bbox, "ymax").text = "5"
    ElementTree.ElementTree(root).write(path)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    samples = []
    for i in range(2):
        img_name = f"{i:06d}.jpg"
        ann_name = f"{i:06d}.xml"
        img_path = os.path.join("Images", "trainval", img_name)
        ann_path = os.path.join("Annotations", "trainval", ann_name)
        _write_image(os.path.join(root, img_path))
        _write_annotation(os.path.join(root, ann_path), img_name, DIOR.classes)
        samples.append(
            {"image_path": img_path, "label_path": ann_path, "split": "train" if i == 0 else "val"}
        )

    test_img_path = os.path.join("Images", "test", "000000.jpg")
    _write_image(os.path.join(root, test_img_path))
    samples.append({"image_path": test_img_path, "label_path": None, "split": "test"})

    pd.DataFrame(samples).to_csv(os.path.join(root, "sample_df.csv"), index=False)
    return root


def test_getitem_train(prepared_root):
    ds = DIOR(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)
    assert "bbox_xyxy" in sample
    assert "label" in sample


def test_getitem_test(prepared_root):
    ds = DIOR(root=prepared_root, split="test", download=False)
    sample = ds[0]
    assert "bbox_xyxy" not in sample


def test_len(prepared_root):
    ds = DIOR(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_invalid_split():
    with pytest.raises(AssertionError):
        DIOR(split="bad")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        DIOR(root=str(tmp_path), download=False)


def test_parse_pascal_voc(prepared_root):
    parsed = parse_pascal_voc(os.path.join(prepared_root, "Annotations", "trainval", "000000.xml"))
    assert parsed["labels"] == [DIOR.classes[0]]


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DIOR(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
