"""Tropical Cyclone Wind Estimation Competition dataset (ported from torchgeo.datasets.cyclone)."""

import os

import numpy as np
import pandas as pd
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import which


class TropicalCyclone(NonGeoDataset):
    """Tropical Cyclone Wind Estimation Competition dataset.

    A collection of tropical storms in the Atlantic and East Pacific Oceans
    from 2000 to 2019 with corresponding maximum sustained surface wind
    speed. This dataset is split into training and test categories for the
    purpose of a competition. Read more about the competition here:
    https://www.drivendata.org/competitions/72/predict-wind-speeds/.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/JSTARS.2020.3011907

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/nasa-tropical-storm-challenge"
    size = 366

    def __init__(self, root="data", split="train", transforms=None, download=False):
        """Initialize a new TropicalCyclone instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert split in {"train", "test"}

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download

        self.filename = f"{split}_set"
        if split == "train":
            self.filename = f"{split}ing_set"

        self._verify()

        self.features = pd.read_csv(os.path.join(root, f"{self.filename}_features.csv"))
        self.labels = pd.read_csv(os.path.join(root, f"{self.filename}_labels.csv"))

    def __getitem__(self, index):
        sample = {
            "relative_time": ops.convert_to_tensor(int(self.features.iat[index, 2])),
            "ocean": ops.convert_to_tensor(int(self.features.iat[index, 3])),
            "label": ops.convert_to_tensor(int(self.labels.iat[index, 1])),
        }

        image_id = str(self.labels.iat[index, 0])
        sample["image"] = self._load_image(image_id)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.labels)

    def _load_image(self, image_id):
        """Load a single image."""
        filename = os.path.join(self.root, self.split, f"{image_id}.jpg")
        with Image.open(filename) as f:
            img = f.convert("RGB")
            if img.height != self.size or img.width != self.size:
                resample = Image.Resampling.BILINEAR
                img = img.resize(size=(self.size, self.size), resample=resample)
            array = np.array(img).astype("float32")
            return ops.convert_to_tensor(array)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        files = [f"{self.filename}_features.csv", f"{self.filename}_labels.csv"]
        exists = [os.path.exists(os.path.join(self.root, file)) for file in files]
        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self):
        """Download the dataset."""
        directory = os.path.join(self.root, self.split)
        os.makedirs(directory, exist_ok=True)
        azcopy = which("azcopy")
        azcopy("sync", f"{self.url}/{self.split}", directory, "--recursive=true")
        files = [f"{self.filename}_features.csv", f"{self.filename}_labels.csv"]
        for file in files:
            azcopy("copy", f"{self.url}/{file}", self.root)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image, label = sample["image"], sample["label"]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = int(ops.convert_to_numpy(sample["prediction"]))

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(ops.convert_to_numpy(image).astype("uint8"))
        ax.axis("off")

        if show_titles:
            title = f"Label: {int(ops.convert_to_numpy(label))}"
            if showing_predictions:
                title += f"\nPrediction: {prediction}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
