import json
import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.vhr10 import VHR10


def _write_data(path, img):
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    Image.fromarray(img).save(path)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    directory = os.path.join(root, "NWPU VHR-10 dataset")
    pos_dir = os.path.join(directory, "positive image set")
    neg_dir = os.path.join(directory, "negative image set")
    os.makedirs(pos_dir)
    os.makedirs(neg_dir)

    n_imgs = 3
    annotations = {"images": [], "annotations": []}
    for img_id in range(1, n_imgs + 1):
        img = np.random.randint(255, size=(8, 8), dtype=np.uint8)
        _write_data(os.path.join(pos_dir, f"00{img_id}.jpg"), img)
        _write_data(os.path.join(neg_dir, f"00{img_id}.jpg"), img)
        annotations["images"].append(
            {"file_name": f"00{img_id}.jpg", "height": 8, "width": 8, "id": img_id - 1}
        )

    for ann_id, image in enumerate(annotations["images"]):
        annotations["annotations"].append(
            {
                "id": ann_id,
                "image_id": image["id"],
                "category_id": 1,
                "area": 4.0,
                "bbox": [4, 4, 2, 2],
                "segmentation": [[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]],
                "iscrowd": 0,
            }
        )

    with open(os.path.join(directory, "annotations.json"), "w") as f:
        json.dump(annotations, f)

    # Touch dummy archive files so _check_integrity's presence checks (when
    # checksum=False) pass without needing real zips.
    open(os.path.join(root, "NWPU VHR-10 dataset.zip"), "w").close()

    return root


def test_getitem_positive(prepared_root):
    ds = VHR10(root=prepared_root, split="positive", download=False, checksum=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)
    assert "label" in sample
    assert "bbox_xyxy" in sample
    assert "mask" in sample


def test_getitem_negative(prepared_root):
    ds = VHR10(root=prepared_root, split="negative", download=False, checksum=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (8, 8, 3)
    assert "label" not in sample


def test_len(prepared_root):
    ds = VHR10(root=prepared_root, split="positive", download=False, checksum=False)
    assert len(ds) == 3


def test_invalid_split():
    with pytest.raises(AssertionError):
        VHR10(split="bogus")


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        VHR10(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = VHR10(root=prepared_root, split="positive", download=False, checksum=False)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()


def test_plot_predictions(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from keras import ops

    ds = VHR10(root=prepared_root, split="positive", download=False, checksum=False)
    sample = ds[0]
    sample["prediction_label"] = sample["label"]
    sample["prediction_bbox_xyxy"] = sample["bbox_xyxy"]
    sample["prediction_mask"] = sample["mask"]
    sample["prediction_score"] = ops.convert_to_tensor(np.array([0.7]))
    ds.plot(sample)
    plt.close()
