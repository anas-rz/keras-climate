import csv
import os

import numpy as np
import pytest
from PIL import Image

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.dlrsd import DLRSD, DLRSDMultilabel

SIZE = 8
CLASSES = ("agricultural", "airplane")


def _build_dlrsd_dir(root):
    folder = os.path.join(root, "DLRSD")
    images_dir = os.path.join(folder, "Images")
    labels_dir = os.path.join(folder, "Labels")

    csv_rows = []
    for cls in CLASSES:
        os.makedirs(os.path.join(images_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, cls), exist_ok=True)
        for i in range(2):
            name = f"{cls}{i:02d}"
            arr = np.random.randint(0, 255, (SIZE, SIZE, 3), dtype="uint8")
            Image.fromarray(arr, mode="RGB").save(os.path.join(images_dir, cls, f"{name}.tif"))

            mask_val = (i % 17) + 1
            mask_arr = np.full((SIZE, SIZE), mask_val, dtype="uint8")
            Image.fromarray(mask_arr).convert("P").save(os.path.join(labels_dir, cls, f"{name}.png"))

            labels = [0] * 17
            labels[mask_val - 1] = 1
            csv_rows.append([name, *labels])

    header = ["image"] + [f"class_{i}" for i in range(17)]
    with open(os.path.join(folder, "multilabels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)

    return folder


@pytest.fixture
def prepared_root(tmp_path):
    _build_dlrsd_dir(str(tmp_path))
    return str(tmp_path)


def test_dlrsd_getitem(prepared_root):
    ds = DLRSD(root=prepared_root, download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (SIZE, SIZE, 3)
    assert tuple(sample["mask"].shape) == (SIZE, SIZE)


def test_dlrsd_len(prepared_root):
    ds = DLRSD(root=prepared_root, download=False)
    assert len(ds) == 4


def test_dlrsd_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        DLRSD(root=str(tmp_path), download=False)


def test_dlrsd_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DLRSD(root=prepared_root, download=False)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()

    sample["prediction"] = sample["mask"]
    ds.plot(sample)
    plt.close()


def test_multilabel_getitem(prepared_root):
    ds = DLRSDMultilabel(root=prepared_root, download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (SIZE, SIZE, 3)
    assert tuple(sample["label"].shape) == (17,)


def test_multilabel_len(prepared_root):
    ds = DLRSDMultilabel(root=prepared_root, download=False)
    assert len(ds) == 4


def test_multilabel_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        DLRSDMultilabel(root=str(tmp_path), download=False)


def test_multilabel_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DLRSDMultilabel(root=prepared_root, download=False)
    sample = ds[0]
    ds.plot(sample, suptitle="Test")
    plt.close()

    sample["prediction"] = sample["label"]
    ds.plot(sample)
    plt.close()
