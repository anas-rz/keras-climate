"""SODA datasets (ported from torchgeo.datasets.soda)."""

import json
import os

import numpy as np
import pandas as pd
from keras import ops
from PIL import Image
from shapely import MultiPoint, Polygon

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_and_extract_archive, download_url


class SODAA(NonGeoDataset):
    """SODA-A dataset.

    The `SODA-A <https://shaunyuan22.github.io/SODA/>`_ dataset is a high resolution
    aerial imagery dataset for small object detection.

    Dataset features:

    * 2513 images
    * 872,069 annotations with oriented bounding boxes
    * 9 classes

    Dataset format:

    * Images are three channel .jpg files.
    * Annotations are in json files

    Classes:

    0. Airplane
    1. Helicopter
    2. Small vehicle
    3. Large vehicle
    4. Ship
    5. Container
    6. Storage tank
    7. Swimming-pool
    8. Windmill
    9. Other

    If you use this dataset in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/10168277
    """

    url = "https://hf.co/datasets/torchgeo/soda-a/resolve/b082b9555ea9960d614b54a8cecde4cc63ec5481/{}"

    files = {
        "images": {
            "filename": "Images.zip",
            "sha256": "90889eae9e6b168ee598576d43f2155cffbe3016209b5f1ce0f5053c2a6eab48",
        },
        "labels": {
            "filename": "Annotations.zip",
            "sha256": "c088d7cc4afafc72509756163efc885e2cad5153161bf4c02b6be37eb4f7d0ff",
        },
    }

    classes = (
        "airplane",
        "helicopter",
        "small-vehicle",
        "large-vehicle",
        "ship",
        "container",
        "storage-tank",
        "swimming-pool",
        "windmill",
        "other",
    )

    valid_splits = ("train", "val", "test")

    valid_orientations = ("oriented", "horizontal")

    def __init__(
        self,
        root="data",
        split="train",
        bbox_orientation="horizontal",
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new instance of SODA-A dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bbox_orientation: one of "oriented" or "horizontal"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum of the downloaded files (may be slow)

        Raises:
            AssertionError: if *split* or *bbox_orientation* argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.valid_splits, f"split must be one of {self.valid_splits}"

        assert bbox_orientation in self.valid_orientations, (
            f"bbox_orientation must be one of {self.valid_orientations}"
        )

        self.root = root
        self.split = split
        self.bbox_orientation = bbox_orientation
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.sample_df = pd.read_csv(os.path.join(self.root, "sample_df.csv"))
        self.sample_df = self.sample_df[
            self.sample_df["split"] == self.split
        ].reset_index(drop=True)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sample_df)

    def __getitem__(self, index):
        """Return the sample at the given index."""
        row = self.sample_df.iloc[index]

        image = self._load_image(os.path.join(self.root, row["image_path"]))
        boxes, labels = self._load_labels(os.path.join(self.root, row["label_path"]))

        sample = {"image": image, "label": labels}

        if self.bbox_orientation == "oriented":
            sample["bbox"] = boxes
        else:
            sample["bbox_xyxy"] = boxes

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path):
        """Load an image from disk.

        Returns:
            the image as a tensor, shape (H, W, C)
        """
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
            return ops.convert_to_tensor(array)

    def _load_labels(self, path):
        """Load labels from disk.

        Returns:
            tuple of:
                - boxes: tensor of bounding boxes in XYXY format [N, 4]
                - labels: tensor of class labels [N]
        """
        with open(path) as f:
            data = json.load(f)

        boxes = []
        labels = []

        for ann in data["annotations"]:
            # Extract polygon points
            coords = ann["poly"]

            points = [
                (coords[i], coords[i + 1])
                for i in range(0, len(coords), 2)
                if i + 1 < len(coords)
            ]

            # Convert to axis-aligned bounding box
            if self.bbox_orientation == "horizontal":
                shapely_poly = Polygon(points)
                minx, miny, maxx, maxy = shapely_poly.bounds
                boxes.append([minx, miny, maxx, maxy])
            else:  # convert to oriented bbox
                hull = MultiPoint(points).convex_hull
                min_rect = hull.minimum_rotated_rectangle
                if isinstance(min_rect, Polygon):
                    rect_coords = list(min_rect.exterior.coords)[:-1]
                    obb_coords = [coord for point in rect_coords for coord in point]
                    boxes.append(obb_coords)
            labels.append(ann["category_id"])

        boxes_tensor = ops.convert_to_tensor(np.array(boxes, dtype="float32"))
        labels_tensor = ops.convert_to_tensor(np.array(labels, dtype="int64"))

        return boxes_tensor, labels_tensor

    def _verify(self):
        """Verify the integrity of the dataset."""
        exists = []
        df_path = os.path.join(self.root, "sample_df.csv")

        if os.path.exists(df_path):
            exists.append(True)
            df = pd.read_csv(df_path)
            df = df[df["split"] == self.split].reset_index(drop=True)
            for idx, row in df.iterrows():
                image_path = os.path.join(self.root, row["image_path"])
                label_path = os.path.join(self.root, row["label_path"])
                exists.append(os.path.exists(image_path) and os.path.exists(label_path))
        else:
            exists.append(False)

        if all(exists):
            return

        exists = []

        for file in self.files.values():
            archive_path = os.path.join(self.root, file["filename"])
            if os.path.exists(archive_path):
                if self.checksum and not check_integrity(
                    archive_path, sha256=file["sha256"]
                ):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the dataset."""
        for file in self.files.values():
            download_and_extract_archive(
                self.url.format(file["filename"]),
                self.root,
                filename=file["filename"],
                sha256=file["sha256"] if self.checksum else None,
            )

        # also download the sample_df
        download_url(
            self.url.format("sample_df.csv"), self.root, filename="sample_df.csv"
        )

    def plot(self, sample, show_titles=True, suptitle=None, box_alpha=0.7):
        """Plot a sample from the dataset with legend."""
        import matplotlib.pyplot as plt
        from matplotlib import patches

        image = ops.convert_to_numpy(sample["image"])
        if self.bbox_orientation == "horizontal":
            boxes = ops.convert_to_numpy(sample["bbox_xyxy"])
        else:
            boxes = ops.convert_to_numpy(sample["bbox"])
        labels = ops.convert_to_numpy(sample["label"])

        fig, ax = plt.subplots(ncols=1, figsize=(10, 10))

        ax.imshow(image)
        ax.axis("off")

        cm = plt.get_cmap("gist_rainbow")

        unique_labels = set()
        legend_elements = []

        for box, label_idx in zip(boxes, labels):
            color = cm(label_idx / len(self.classes))
            label = self.classes[label_idx]

            if self.bbox_orientation == "horizontal":
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
                ax.add_patch(rect)
            else:
                # Oriented box: [x1,y1,x2,y2,x3,y3,x4,y4]
                vertices = box.reshape(4, 2)
                polygon = patches.Polygon(
                    vertices,
                    linewidth=2,
                    alpha=box_alpha,
                    linestyle="solid",
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(polygon)

            if label not in unique_labels:
                legend_elements.append(
                    patches.Patch(facecolor=color, alpha=box_alpha, label=label)
                )
                unique_labels.add(label)

        ax.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=len(legend_elements),
            mode="expand",
        )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
