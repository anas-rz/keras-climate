import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.fair1m import FAIR1M

VOC_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <source>
    <filename>{filename}</filename>
    <origin>GF2/GF3</origin>
  </source>
  <objects>
    <object>
      <coordinate>pixel</coordinate>
      <type>rectangle</type>
      <description>None</description>
      <possibleresult>
        <name>Small Car</name>
      </possibleresult>
      <points>
        <point>1.0,1.0</point>
        <point>1.0,5.0</point>
        <point>5.0,5.0</point>
        <point>5.0,1.0</point>
        <point>1.0,1.0</point>
      </points>
    </object>
  </objects>
</annotation>
"""


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)

    train_img_dir = os.path.join(root, "train", "part1", "images")
    train_lbl_dir = os.path.join(root, "train", "part1", "labelXml")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    # part2 directories must also exist for _verify() to pass, even if empty
    os.makedirs(os.path.join(root, "train", "part2", "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "train", "part2", "labelXml"), exist_ok=True)

    image = Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype="uint8"))
    image.save(os.path.join(train_img_dir, "1.tif"))
    with open(os.path.join(train_lbl_dir, "1.xml"), "w") as f:
        f.write(VOC_XML.format(filename="1.tif"))

    test_img_dir = os.path.join(root, "test", "images")
    os.makedirs(test_img_dir, exist_ok=True)
    image.save(os.path.join(test_img_dir, "2.tif"))

    return root


def test_getitem_train(prepared_root):
    ds = FAIR1M(root=prepared_root, split="train", download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert "bbox_xyxy" in sample
    assert "label" in sample


def test_getitem_test(prepared_root):
    ds = FAIR1M(root=prepared_root, split="test", download=False)
    sample = ds[0]
    assert "image" in sample
    assert "bbox_xyxy" not in sample


def test_len(prepared_root):
    ds = FAIR1M(root=prepared_root, split="train", download=False)
    assert len(ds) == 1


def test_invalid_split(prepared_root):
    with pytest.raises(AssertionError):
        FAIR1M(root=prepared_root, split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        FAIR1M(root=str(tmp_path), split="train", download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = FAIR1M(root=prepared_root, split="train", download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
