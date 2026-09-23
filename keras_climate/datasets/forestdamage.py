"""Forest Damage dataset (ported from torchgeo.datasets.forestdamage)."""

import glob
import os
from xml.etree import ElementTree

import numpy as np
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_and_extract_archive, extract_archive


def parse_pascal_voc(path):
    """Read a PASCAL VOC annotation file.

    Returns a dict of image filename, bounding boxes, and class labels.
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

        label_var = obj.find("damage")
        if label_var is not None:
            label = label_var.text
        else:
            label = "other"
        bboxes.append(bbox)
        labels.append(label)
    return {"filename": filename, "bboxes": bboxes, "labels": labels}


class ForestDamage(NonGeoDataset):
    """Forest Damage dataset.

    The `ForestDamage
    <https://lila.science/datasets/forest-damages-larch-casebearer/>`_
    dataset contains drone imagery that can be used for tree identification,
    as well as tree damage classification for larch trees.

    Dataset features:

    * 1543 images
    * 101,878 tree annotations
    * subset of 840 images contain 44,522 annotations about tree health
      (Healthy (H), Light Damage (LD), High Damage (HD)), all other images
      have "other" as damage level

    Dataset format:

    * images are three-channel jpgs
    * annotations are in Pascal VOC XML format

    Dataset classes:

    0. other
    1. healthy
    2. light damage
    3. high damage

    If the download fails or stalls, it is recommended to try azcopy as
    suggested `here <https://lila.science/faq>`__. It is expected that the
    downloaded data file with name ``Data_Set_Larch_Casebearer`` can be found
    in ``root``.

    If you use this dataset in your research, please cite:

    * Swedish Forest Agency (2021): Forest Damages - Larch Casebearer 1.0.
      National Forest Data Lab. Dataset.
    """

    classes = ("other", "H", "LD", "HD")
    url = "https://lilablobssc.blob.core.windows.net/larch-casebearer/Data_Set_Larch_Casebearer.zip"
    data_dir = "Data_Set_Larch_Casebearer"
    md5 = "907815bcc739bff89496fac8f8ce63d7"

    def __init__(self, root="data", transforms=None, download=False, checksum=True):
        """Initialize a new ForestDamage dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        self._verify()

        self.files = self._load_files(self.root)

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, index):
        """Return an index within the dataset."""
        files = self.files[index]
        parsed = parse_pascal_voc(files["annotation"])
        image = self._load_image(files["image"])

        boxes, labels = self._load_target(parsed["bboxes"], parsed["labels"])

        sample = {"image": image, "bbox_xyxy": boxes, "label": labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def _load_files(self, root):
        """Return the paths of the files in the dataset."""
        images = sorted(glob.glob(os.path.join(root, self.data_dir, "**", "Images", "*.JPG")))
        annotations = sorted(
            glob.glob(os.path.join(root, self.data_dir, "**", "Annotations", "*.xml"))
        )

        files = [
            {"image": image, "annotation": annotation}
            for image, annotation in zip(images, annotations)
        ]

        return files

    def _load_image(self, path):
        """Load a single image (returned channels-last as HxWxC)."""
        from PIL import Image

        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
            return ops.convert_to_tensor(array)

    def _load_target(self, bboxes, labels_list):
        """Load the target boxes and labels for a single image."""
        labels = ops.convert_to_tensor(
            np.array([self.class_to_idx[label] for label in labels_list], dtype="int64")
        )
        boxes = ops.convert_to_tensor(np.array(bboxes, dtype="float32"))
        return boxes, labels

    def _verify(self):
        """Verify the integrity of the dataset."""
        filepath = os.path.join(self.root, self.data_dir)
        if os.path.isdir(filepath):
            return

        filepath = os.path.join(self.root, f"{self.data_dir}.zip")
        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # else download the dataset
        self._download()

    def _download(self):
        """Download the dataset and extract it."""
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.data_dir + ".zip",
            md5=self.md5 if self.checksum else None,
        )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt
        from matplotlib import patches

        image = ops.convert_to_numpy(sample["image"])

        ncols = 1
        showing_predictions = "prediction_bbox_xyxy" in sample
        if showing_predictions:
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if not showing_predictions:
            axs = [axs]

        axs[0].imshow(image)
        axs[0].axis("off")

        bboxes = [
            patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            for bbox in ops.convert_to_numpy(sample["bbox_xyxy"])
        ]
        for bbox in bboxes:
            axs[0].add_patch(bbox)

        if show_titles:
            axs[0].set_title("Ground Truth")

        if showing_predictions:
            axs[1].imshow(image)
            axs[1].axis("off")

            pred_bboxes = [
                patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                )
                for bbox in ops.convert_to_numpy(sample["prediction_bbox_xyxy"])
            ]
            for bbox in pred_bboxes:
                axs[1].add_patch(bbox)

            if show_titles:
                axs[1].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
