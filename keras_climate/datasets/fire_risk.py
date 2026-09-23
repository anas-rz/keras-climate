"""FireRisk dataset (ported from torchgeo.datasets.fire_risk)."""

import os

from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoClassificationDataset
from .utils import download_url, extract_archive


class FireRisk(NonGeoClassificationDataset):
    """FireRisk dataset.

    The `FireRisk <https://github.com/CharmonyShen/FireRisk>`__ dataset is a
    dataset for remote sensing fire risk classification.

    Dataset features:

    * 91,872 images with 1 m per pixel resolution (320x320 px)
    * 70,331 and 21,541 train and val images, respectively
    * three spectral bands - RGB
    * 7 fire risk classes
    * images extracted from NAIP tiles

    Dataset format:

    * images are three-channel pngs

    Dataset classes:

    0. high
    1. low
    2. moderate
    3. non-burnable
    4. very_high
    5. very_low
    6. water

    If you use this dataset in your research, please cite:

    * https://arxiv.org/abs/2303.07035
    """

    url = "https://hf.co/datasets/isaaccorley/fire_risk/resolve/e6046a04350c6f1ab4ad791fb3a40bf8940be269/FireRisk.zip"
    sha256 = "80d51d5bf5004e4cfffac5a64a23b46c4fb3427ebd1a4f0ac848ca2dbbbf46ac"
    filename = "FireRisk.zip"
    directory = "FireRisk"
    splits = ("train", "val")
    classes = (
        "High",
        "Low",
        "Moderate",
        "Non-burnable",
        "Very_High",
        "Very_Low",
        "Water",
    )

    def __init__(
        self,
        root="data",
        split="train",
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new FireRisk dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "val"
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
        assert split in self.splits
        self.root = root
        self.split = split
        self.download = download
        self.checksum = checksum
        self._verify()

        super().__init__(root=os.path.join(root, self.directory, self.split), transforms=transforms)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        path = os.path.join(self.root, self.directory)
        if os.path.exists(path):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self):
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            sha256=self.sha256 if self.checksum else None,
        )

    def _extract(self):
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        label = int(ops.convert_to_numpy(sample["label"]))
        label_class = self.classes[label]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = int(ops.convert_to_numpy(sample["prediction"]))
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {label_class}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_class}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
