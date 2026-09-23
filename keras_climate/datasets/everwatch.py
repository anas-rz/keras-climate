"""EverWatch dataset (ported from torchgeo.datasets.everwatch)."""

import os

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_and_extract_archive, extract_archive


class EverWatch(NonGeoDataset):
    """EverWatch Bird Detection dataset.

    The `EverWatch Bird Detection <https://zenodo.org/records/11165946>`__
    dataset contains high-resolution aerial images of birds in the Everglades
    National Park. Seven bird species have been annotated and classified.

    Dataset format:

    * images are three-channel pngs
    * annotations are csv files

    Dataset classes:

    0. White Ibis (Eudocimus albus)
    1. Great Egret (Ardea alba)
    2. Great Blue Heron (Ardea herodias)
    3. Snowy Egret (Egretta thula)
    4. Wood Stork (Mycteria americana)
    5. Roseate Spoonbill (Platalea ajaja)
    6. Anhinga (Anhinga anhinga)
    7. Unknown White (only present in test split)

    If you use this dataset in your research, please cite:

    * https://doi.org/10.5281/zenodo.11165946
    """

    dir = "everwatch-benchmark"

    url = "https://zenodo.org/records/11165946/files/everwatch-benchmark.zip?download=1"

    md5 = "ab34b0873b659656e36a9b41648f98db"

    zipfilename = "everwatch-benchmark.zip"

    valid_splits = ("train", "test")

    classes = (
        "White Ibis",
        "Great Egret",
        "Great Blue Heron",
        "Snowy Egret",
        "Wood Stork",
        "Roseate Spoonbill",
        "Anhinga",
        "Unknown White",  # only present in test split
    )

    def __init__(
        self,
        root="data",
        split="train",
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new EverWatch dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of {"train", "test"} to specify the dataset split
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
            AssertionError: If *split* argument is invalid.
        """
        assert split in self.valid_splits, (
            f"Split '{split}' not supported, please use one of {self.valid_splits}"
        )

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        self._verify()

        self.annot_df = pd.read_csv(os.path.join(self.root, self.dir, f"{self.split}.csv"))

        # remove all entries where xmin == xmax or ymin == ymax
        self.annot_df = self.annot_df[
            (self.annot_df["xmin"] != self.annot_df["xmax"])
            & (self.annot_df["ymin"] != self.annot_df["ymax"])
        ].reset_index(drop=True)

        # group per image path to get all annotations for one sample
        self.annot_df["sample_index"] = pd.factorize(self.annot_df["image_path"])[0]
        self.annot_df = self.annot_df.set_index(["sample_index", self.annot_df.index])

        self.class2idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        """Return the number of samples in the dataset."""
        index = self.annot_df.index
        return len(index.levels[0])

    def __getitem__(self, index):
        """Return an index within the dataset."""
        sample_df = self.annot_df.loc[index]

        img_path = os.path.join(self.root, self.dir, sample_df["image_path"].iloc[0])

        image = self._load_image(img_path)

        boxes, labels = self._load_target(sample_df)

        sample = {"image": image, "bbox_xyxy": boxes, "label": labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path):
        """Load a single image (returned channels-last as HxWxC)."""
        from PIL import Image

        with Image.open(path) as img:
            array = np.array(img)
            return ops.convert_to_tensor(array)

    def _load_target(self, sample_df):
        """Load target boxes and labels from a dataframe row group."""
        boxes = ops.convert_to_tensor(
            sample_df[["xmin", "ymin", "xmax", "ymax"]].values.astype("float32")
        )
        labels = ops.convert_to_tensor(
            np.array(
                [self.class2idx[label] for label in sample_df["label"].tolist()],
                dtype="int64",
            )
        )
        return boxes, labels

    def _verify(self):
        """Verify the integrity of the dataset."""
        exists = []
        df_path = os.path.join(self.root, self.dir, f"{self.split}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            image_paths = df["image_path"].unique().tolist()
            for path in image_paths:
                if os.path.exists(os.path.join(self.root, self.dir, path)):
                    exists.append(True)
        else:
            exists.append(False)

        if all(exists):
            return

        filepath = os.path.join(self.root, self.zipfilename)
        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the dataset and extract it."""
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.zipfilename,
            md5=self.md5 if self.checksum else None,
        )

    def plot(self, sample, show_titles=True, suptitle=None, box_alpha=0.7):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt
        from matplotlib import patches

        image = ops.convert_to_numpy(sample["image"])
        boxes = ops.convert_to_numpy(sample["bbox_xyxy"])
        labels = ops.convert_to_numpy(sample["label"])

        fig, axs = plt.subplots(ncols=1, figsize=(10, 10))

        axs.imshow(image)
        axs.axis("off")

        cm = plt.get_cmap("gist_rainbow")

        for box, label_idx in zip(boxes, labels):
            color = cm(label_idx / len(self.classes))
            label = self.classes[label_idx]

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
            if show_titles:
                axs.text(
                    x1,
                    y1 - 5,
                    label,
                    color="white",
                    fontsize=8,
                    bbox={"facecolor": color, "alpha": box_alpha},
                )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
