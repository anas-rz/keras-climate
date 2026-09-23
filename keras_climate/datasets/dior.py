"""DIOR dataset (ported from torchgeo.datasets.dior)."""

import os
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_and_extract_archive, download_url, extract_archive


def parse_pascal_voc(path):
    """Read a PASCAL VOC annotation file.

    Returns a dict of image filename, bounding box coords, and class labels.
    """
    et = ElementTree.parse(path)
    element = et.getroot()
    filename = element.find("filename").text
    labels, bboxes = [], []

    for obj in element.findall("object"):
        bndbox = obj.find("bndbox")
        bbox = [
            int(bndbox.find("xmin").text),
            int(bndbox.find("ymin").text),
            int(bndbox.find("xmax").text),
            int(bndbox.find("ymax").text),
        ]
        label = obj.find("name").text
        bboxes.append(bbox)
        labels.append(label)

    return {"filename": filename, "bboxes": bboxes, "labels": labels}


class DIOR(NonGeoDataset):
    """DIOR dataset.

    `DIOR <https://arxiv.org/abs/1909.00133>`__ dataset contains horizontal
    bounding box annotations of Google Earth Aerial RGB imagery. The test
    split does not contain bounding box annotations and labels.

    Dataset features:

    * 20 classes
    * 192,472 manually annotated bounding box instances

    Dataset format:

    * Images are three channel .jpg files.
    * Annotations are in `Pascal VOC XML format
      <https://roboflow.com/formats/pascal-voc-xml>`_

    Classes:

    0. Airplane
    1. Airport
    2. Baseball Field
    3. Basketball Court
    4. Bridge
    5. Chimney
    6. Dam
    7. Expressway Service Area
    8. Expressway Toll Station
    9. Golf Field
    10. Ground Track Field
    11. Harbor
    12. Overpass
    13. Ship
    14. Stadium
    15. Storage Tank
    16. Tennis Court
    17. Train Station
    18. Vehicle
    19. Windmill

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1909.00133
    """

    url = "https://hf.co/datasets/torchgeo/dior/resolve/0b8439a538b66457e553a8bd9105ef093cce8169/{}"

    files = {
        "trainval": {
            "images": {
                "filename": "Images_trainval.zip",
                "sha256": "f824dbd8152de43c0e0151f13d892dcc69ce2c3d18b952507cc843cc8ce85d27",
            },
            "labels": {
                "filename": "Annotations_trainval.zip",
                "sha256": "e688da8179324f6d1d0e84b54ed75b0922e45ce30817c4973f1c5b1efd03a8e4",
            },
        },
        "test": {
            "images": {
                "filename": "Images_test.zip",
                "sha256": "a643bcdbaf3e8f9d9847a6ae023c2b624ae1e1dea278b2ca1b5ee90759426f1d",
            }
        },
    }

    valid_splits = ("train", "val", "test")

    classes = (
        "airplane",
        "airport",
        "baseballfield",
        "basketballcourt",
        "bridge",
        "chimney",
        "dam",
        "expresswayservicearea",
        "expresswaytollstation",
        "golffield",
        "groundtrackfield",
        "harbor",
        "overpass",
        "ship",
        "stadium",
        "storagetank",
        "tenniscourt",
        "trainstation",
        "vehicle",
        "windmill",
    )

    def __init__(self, root="data", split="train", transforms=None, download=False, checksum=True):
        """Initialize a new DIOR dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split of the dataset to use, one of 'train', 'val', 'test'
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found or corrupted and
                *download* is False.
            AssertionError: If *split* argument is invalid.
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        assert split in self.valid_splits, f"Split must be one of {self.valid_splits}."
        self.split = split

        self._verify()

        self.sample_df = pd.read_csv(os.path.join(self.root, "sample_df.csv"))

        self.sample_df = self.sample_df[self.sample_df["split"] == self.split].reset_index(drop=True)

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.sample_df)

    def __getitem__(self, index):
        row = self.sample_df.iloc[index]

        image = self._load_image(os.path.join(self.root, row["image_path"]))

        sample = {"image": image}

        if self.split != "test":
            boxes, labels = self._load_target(os.path.join(self.root, row["label_path"]))
            sample["bbox_xyxy"] = boxes
            sample["label"] = labels

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path):
        """Load a single image."""
        with Image.open(path) as img:
            array = np.array(img.convert("RGB")).astype("float32")
            return ops.convert_to_tensor(array)

    def _load_target(self, path):
        """Load the target bounding boxes and labels for a single image."""
        parsed = parse_pascal_voc(path)
        boxes = ops.convert_to_tensor(np.array(parsed["bboxes"], dtype="float32"))
        labels = ops.convert_to_tensor(
            np.array([self.class_to_idx[label] for label in parsed["labels"]], dtype="int64")
        )
        return boxes, labels

    def _verify(self):
        """Verify the integrity of the dataset."""
        df_path = os.path.join(self.root, "sample_df.csv")
        exists = []
        if os.path.exists(df_path):
            exists.append(True)
            df = pd.read_csv(df_path)
            df = df[df["split"] == self.split].reset_index(drop=True)
            for idx, row in df.iterrows():
                if os.path.exists(os.path.join(self.root, row["image_path"])):
                    exists.append(True)
                else:
                    exists.append(False)
        else:
            exists.append(False)

        if all(exists):
            return

        exists = []
        if self.split in ["train", "val"]:
            files = self.files["trainval"]
        else:
            files = self.files["test"]

        for key in files:
            filename = files[key]["filename"]
            sha256 = files[key]["sha256"]
            path = os.path.join(self.root, filename)
            if os.path.exists(path):
                if self.checksum and not check_integrity(path, sha256=sha256):
                    raise RuntimeError("Dataset found, but corrupted.")
                extract_archive(path)
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the dataset and extract it."""
        if self.split in ["train", "val"]:
            files = self.files["trainval"]
        else:
            files = self.files["test"]

        for key in files:
            filename = files[key]["filename"]
            sha256 = files[key]["sha256"]
            download_and_extract_archive(
                self.url.format(filename), self.root, filename=filename, sha256=sha256 if self.checksum else None
            )

        # download the sample_df.csv file
        download_url(self.url.format("sample_df.csv"), self.root, filename="sample_df.csv")

    def plot(self, sample, show_titles=True, suptitle=None, box_alpha=0.7):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt
        from matplotlib import patches

        image = ops.convert_to_numpy(sample["image"]).astype("uint8")
        boxes = ops.convert_to_numpy(sample["bbox_xyxy"])
        labels = ops.convert_to_numpy(sample["label"])

        fig, axs = plt.subplots(ncols=1, figsize=(10, 10))

        axs.imshow(image)
        axs.axis("off")

        cm = plt.get_cmap("gist_rainbow")

        for box, label_idx in zip(boxes, labels):
            color = cm(label_idx / len(self.classes))
            label = self.classes[label_idx]

            # Horizontal box: [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=box_alpha,
                linestyle="solid",
                edgecolor=color,
                facecolor="none",
            )
            axs.add_patch(rect)
            # Add label above box
            axs.text(
                x1, y1 - 5, label, color="white", fontsize=8, bbox={"facecolor": color, "alpha": box_alpha}
            )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
