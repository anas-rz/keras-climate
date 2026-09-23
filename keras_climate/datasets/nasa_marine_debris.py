"""NASA Marine Debris dataset (ported from torchgeo.datasets.nasa_marine_debris)."""

import glob
import os

import numpy as np
import rasterio
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import which


class NASAMarineDebris(NonGeoDataset):
    """NASA Marine Debris dataset.

    The `NASA Marine Debris <https://source.coop/nasa/marine-debris>`__
    dataset is a dataset for detection of floating marine debris in
    satellite imagery.

    Dataset features:

    * 707 patches with 3 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 1 object class: marine_debris
    * images taken by Planet Labs PlanetScope satellites
    * imagery taken from 2016-2019 from coasts of Greece, Honduras, and Ghana

    Dataset format:

    * images are three-channel geotiffs in uint8 format
    * labels are numpy files (.npy) containing bounding box (xyxy) coordinates
    * additional: images in jpg format and labels in geojson format

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.34911/rdnt.9r6ekg

    .. note::

       This dataset requires the following additional library to be
       installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to
         download the dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/nasa-marine-debris"

    def __init__(self, root="data", transforms=None, download=False):
        """Initialize a new NASA Marine Debris Dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.root = root
        self.transforms = transforms
        self.download = download

        self._verify()

        self.source = sorted(glob.glob(os.path.join(self.root, "source", "*.tif")))
        self.labels = sorted(glob.glob(os.path.join(self.root, "labels", "*.npy")))

    def __getitem__(self, index):
        """Return an index within the dataset."""
        with rasterio.open(self.source[index]) as source:
            image = source.read()

        # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0)).astype("float32")
        image = ops.convert_to_tensor(image)

        labels = np.load(self.labels[index])

        # Boxes contain unnecessary value of 1 after xyxy coords
        boxes = labels[:, :4].astype("float32")

        # Filter invalid boxes
        w_check = (boxes[:, 2] - boxes[:, 0]) > 0
        h_check = (boxes[:, 3] - boxes[:, 1]) > 0
        keep = w_check & h_check
        boxes = boxes[keep]

        sample = {"image": image, "bbox_xyxy": ops.convert_to_tensor(boxes)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.source)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the directories already exist
        dirs = ["source", "labels"]
        exists = [os.path.exists(os.path.join(self.root, d)) for d in dirs]
        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self):
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)
        azcopy = which("azcopy")
        azcopy("sync", self.url, self.root, "--recursive=true")

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        ncols = 1

        image = ops.convert_to_numpy(sample["image"]).astype("uint8")
        boxes = ops.convert_to_numpy(sample.get("bbox_xyxy", []))

        showing_predictions = "prediction_bbox_xyxy" in sample
        if showing_predictions:
            ncols += 1
            pred_boxes = ops.convert_to_numpy(sample["prediction_bbox_xyxy"])

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs_list = axs if ncols > 1 else [axs]

        axs_list[0].imshow(image)
        axs_list[0].axis("off")
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
            )
            axs_list[0].add_patch(rect)
        if show_titles:
            axs_list[0].set_title("Ground Truth")

        if showing_predictions:
            axs_list[1].imshow(image)
            axs_list[1].axis("off")
            for box in pred_boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                axs_list[1].add_patch(rect)
            if show_titles:
                axs_list[1].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
