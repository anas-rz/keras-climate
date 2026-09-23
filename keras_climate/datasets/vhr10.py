"""NWPU VHR-10 dataset (ported from torchgeo.datasets.vhr10)."""

import glob
import json
import os
from collections import defaultdict

import numpy as np
from keras import ops
from PIL import Image
from rasterio.features import rasterize
from shapely import Polygon

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_and_extract_archive, download_url, quantile_normalization


class VHR10(NonGeoDataset):
    """NWPU VHR-10 dataset.

    Northwestern Polytechnical University (NWPU) very-high-resolution
    ten-class (VHR-10) remote sensing image dataset.

    Consists of 800 VHR optical remote sensing images, where 715 color
    images were acquired from Google Earth with the spatial resolution
    ranging from 0.5 to 2 m, and 85 pansharpened color infrared (CIR) images
    were acquired from Vaihingen data with a spatial resolution of 0.08 m.

    The data set is divided into two sets:

    * Positive image set (650 images) which contains at least one target
    * Negative image set (150 images) does not contain any targets

    The positive image set consists of objects from ten classes: airplane,
    ship, storage tank, baseball diamond, tennis court, basketball court,
    ground track field, harbor, bridge, and vehicle.

    Includes object detection bounding boxes from the original paper and
    instance segmentation masks from follow-up publications. If you use
    this dataset in your research, please cite:

    * https://doi.org/10.1016/j.isprsjprs.2014.10.002
    * https://doi.org/10.1109/IGARSS.2019.8898573
    * https://doi.org/10.3390/rs12060989
    """

    image_meta = {
        "url": "https://hf.co/datasets/isaaccorley/vhr10/resolve/60ecc4be33609184e2224606858cd00b7daba8df/NWPU%20VHR-10%20dataset.zip",
        "filename": "NWPU VHR-10 dataset.zip",
        "sha256": "3e8c0299bad6b5d2b4d4034095c3581f50a02bc0dcb97fca70f6ad739f7cbf53",
    }
    target_meta = {
        "url": "https://hf.co/datasets/isaaccorley/vhr10/resolve/7e7968ad265dadc4494e0ca4a079e0b63dc6f3f8/annotations.json",
        "filename": "annotations.json",
        "sha256": "dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e",
    }

    categories = (
        "background",
        "airplane",
        "ship",
        "storage tank",
        "baseball diamond",
        "tennis court",
        "basketball court",
        "ground track field",
        "harbor",
        "bridge",
        "vehicle",
    )

    def __init__(self, root="data", split="positive", transforms=None, download=False, checksum=True):
        """Initialize a new VHR-10 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "positive" or "negative"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert split in {"positive", "negative"}

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        directory = os.path.join(self.root, "NWPU VHR-10 dataset", f"{self.split} image set")
        self.files = sorted(glob.glob(os.path.join(directory, "*.jpg")))

        if not self.files:
            raise DatasetNotFoundError(self)

        if split == "positive":
            path = os.path.join(self.root, "NWPU VHR-10 dataset", "annotations.json")
            with open(path) as f:
                annotations = json.load(f)

                # Gather image shapes
                out_shapes = []
                image_ids = {}
                for image in annotations["images"]:
                    out_shapes.append((image["height"], image["width"]))
                    image_ids[image["file_name"]] = image["id"]
                self.ids = [image_ids[os.path.basename(file)] for file in self.files]

                self.labels = defaultdict(list)
                self.boxes = defaultdict(list)
                self.masks = defaultdict(list)
                for annotation in annotations["annotations"]:
                    i = annotation["image_id"]
                    self.labels[i].append(annotation["category_id"])

                    # Convert box format
                    x1, y1, w, h = annotation["bbox"]
                    self.boxes[i].append([x1, y1, x1 + w, y1 + h])

                    # Rasterize segmentation mask
                    segmentation = annotation["segmentation"]  # [[x1, y1, x2, y2, ...]]
                    xs = segmentation[0][::2]  # [x1, x2, ...]
                    ys = segmentation[0][1::2]  # [y1, y2, ...]
                    coords = list(zip(xs, ys))  # [(x1, y1), (x2, y2), ...]
                    shapes = [(Polygon(coords), 1)]
                    mask = rasterize(shapes, out_shapes[i], dtype=np.uint8)
                    self.masks[i].append(mask)

    def __getitem__(self, index):
        sample = {}

        with Image.open(self.files[index]) as f:
            array = np.array(f).astype("float32")
            sample["image"] = ops.convert_to_tensor(array)

        # Only 'positive' split has target labels
        if self.split == "positive":
            id_ = self.ids[index]
            sample["label"] = ops.convert_to_tensor(np.array(self.labels[id_], dtype="int64"))
            sample["bbox_xyxy"] = ops.convert_to_tensor(np.array(self.boxes[id_], dtype="float32"))
            sample["mask"] = ops.convert_to_tensor(np.stack(self.masks[id_]))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _check_integrity(self):
        """Check integrity of dataset."""
        image = check_integrity(
            os.path.join(self.root, self.image_meta["filename"]),
            sha256=self.image_meta["sha256"] if self.checksum else None,
        )

        # Annotations only needed for "positive" image set
        target = True
        if self.split == "positive":
            target = check_integrity(
                os.path.join(self.root, "NWPU VHR-10 dataset", self.target_meta["filename"]),
                sha256=self.target_meta["sha256"] if self.checksum else None,
            )

        return image and target

    def _download(self):
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        # Download images
        download_and_extract_archive(
            self.image_meta["url"],
            self.root,
            filename=self.image_meta["filename"],
            sha256=self.image_meta["sha256"] if self.checksum else None,
        )

        # Annotations only needed for "positive" image set
        if self.split == "positive":
            download_url(
                self.target_meta["url"],
                os.path.join(self.root, "NWPU VHR-10 dataset"),
                self.target_meta["filename"],
                sha256=self.target_meta["sha256"] if self.checksum else None,
            )

    def plot(self, sample, show_titles=True, suptitle=None, show_feats="both", box_alpha=0.7, mask_alpha=0.7):
        """Plot a sample from the dataset.

        Raises:
            AssertionError: if ``show_feats`` argument is invalid
        """
        assert show_feats in {"boxes", "masks", "both"}

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        cm = plt.get_cmap("gist_rainbow")
        ncols = 2 if "prediction_label" in sample else 1
        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 10, 10))

        image = quantile_normalization(sample["image"])
        image = ops.convert_to_numpy(image)
        axs[0, 0].imshow(image)
        axs[0, 0].axis("off")

        if "label" in sample:
            labels = ops.convert_to_numpy(sample["label"])
            boxes = ops.convert_to_numpy(sample["bbox_xyxy"])
            masks = ops.convert_to_numpy(sample["mask"])
            for i in range(len(labels)):
                class_num = int(labels[i])
                color = cm(class_num / len(self.categories))

                if show_feats in {"boxes", "both"}:
                    x1, y1, x2, y2 = boxes[i]
                    r = Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor="none",
                    )
                    axs[0, 0].add_patch(r)

                    label = self.categories[class_num]
                    axs[0, 0].text(x1, y1 - 8, label, color="white", size=11, backgroundcolor="none")

                if show_feats in {"masks", "both"}:
                    mask = masks[i]
                    alpha = mask * mask_alpha
                    mask = mask * class_num
                    axs[0, 0].imshow(mask, cmap=cm, vmin=0, vmax=10, alpha=alpha)

            if show_titles:
                axs[0, 0].set_title("Ground Truth")

        if "prediction_label" in sample:
            axs[0, 1].imshow(image)
            axs[0, 1].axis("off")

            scores = ops.convert_to_numpy(sample["prediction_score"])
            labels = ops.convert_to_numpy(sample["prediction_label"])
            boxes = ops.convert_to_numpy(sample["prediction_bbox_xyxy"])
            for i in range(len(labels)):
                score = scores[i]
                if score < 0.5:
                    continue

                class_num = int(labels[i])
                color = cm(class_num / len(self.categories))

                if show_feats in {"boxes", "both"}:
                    x1, y1, x2, y2 = boxes[i]
                    r = Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        alpha=box_alpha,
                        linestyle="dashed",
                        edgecolor=color,
                        facecolor="none",
                    )
                    axs[0, 1].add_patch(r)

                    label = self.categories[class_num]
                    caption = f"{label} {score:.3f}"
                    axs[0, 1].text(x1, y1 - 8, caption, color="white", size=11, backgroundcolor="none")

                if "prediction_mask" in sample and show_feats in {"masks", "both"}:
                    masks = ops.convert_to_numpy(sample["prediction_mask"])
                    mask = masks[i]
                    alpha = mask * mask_alpha
                    mask = mask * class_num
                    axs[0, 1].imshow(mask, cmap=cm, vmin=0, vmax=10, alpha=alpha)

            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
