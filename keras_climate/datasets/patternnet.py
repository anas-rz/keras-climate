"""PatternNet dataset (ported from torchgeo.datasets.patternnet)."""

import os

from .errors import DatasetNotFoundError
from .geo import NonGeoClassificationDataset
from .utils import download_url, extract_archive


class PatternNet(NonGeoClassificationDataset):
    """PatternNet dataset.

    The `PatternNet <https://sites.google.com/view/zhouwx/dataset>`__
    dataset is a dataset for remote sensing scene classification and image
    retrieval.

    Dataset features:

    * 30,400 images with 6-50 cm per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 38 scene classes, 800 images per class

    Dataset format:

    * images are three-channel jpgs

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1016/j.isprsjprs.2018.01.004
    """

    url = "https://hf.co/datasets/torchgeo/PatternNet/resolve/2dbd901b00e301967a5c5146b25454f5d3455ad0/PatternNet.zip"
    sha256 = "456dd031950b0429518b8ec7d30a5c4b3f6456261ee1dd573a3eee43efb29958"
    filename = "PatternNet.zip"
    directory = os.path.join("PatternNet", "images")

    def __init__(self, root="data", transforms=None, download=False, checksum=True):
        """Initialize a new PatternNet dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.root = root
        self.download = download
        self.checksum = checksum
        self._verify()
        super().__init__(root=os.path.join(root, self.directory), transforms=transforms)

    def _verify(self):
        """Verify the integrity of the dataset."""
        filepath = os.path.join(self.root, self.directory)
        if os.path.exists(filepath):
            return

        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

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
        from keras import ops

        image = ops.convert_to_numpy(sample["image"]).astype("uint8")
        label = int(ops.convert_to_numpy(sample["label"]))

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = int(ops.convert_to_numpy(sample["prediction"]))

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            title = f"Label: {self.classes[label]}"
            if showing_predictions:
                title += f"\nPrediction: {self.classes[prediction]}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
