import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.forestdamage import ForestDamage

VOC_XML = """<?xml version="1.0"?>
<annotation>
  <filename>{filename}</filename>
  <object>
    <damage>H</damage>
    <bndbox>
      <xmin>1</xmin>
      <ymin>1</ymin>
      <xmax>5</xmax>
      <ymax>5</ymax>
    </bndbox>
  </object>
  <object>
    <bndbox>
      <xmin>2</xmin>
      <ymin>2</ymin>
      <xmax>6</xmax>
      <ymax>6</ymax>
    </bndbox>
  </object>
</annotation>
"""


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    region_dir = os.path.join(root, ForestDamage.data_dir, "region1")
    img_dir = os.path.join(region_dir, "Images")
    ann_dir = os.path.join(region_dir, "Annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    image = Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype="uint8"))
    image.save(os.path.join(img_dir, "1.JPG"))
    with open(os.path.join(ann_dir, "1.xml"), "w") as f:
        f.write(VOC_XML.format(filename="1.JPG"))

    return root


def test_getitem(prepared_root):
    ds = ForestDamage(root=prepared_root, download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (16, 16, 3)
    assert tuple(sample["bbox_xyxy"].shape) == (2, 4)
    assert tuple(sample["label"].shape) == (2,)


def test_len(prepared_root):
    ds = ForestDamage(root=prepared_root, download=False)
    assert len(ds) == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        ForestDamage(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ForestDamage(root=prepared_root, download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
