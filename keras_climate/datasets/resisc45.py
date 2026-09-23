"""RESISC45 dataset (ported from torchgeo.datasets.resisc45)."""

import os

from .errors import DatasetNotFoundError
from .geo import NonGeoClassificationDataset
from .utils import download_url, extract_archive


class RESISC45(NonGeoClassificationDataset):
    """NWPU-RESISC45 dataset.

    The `RESISC45 <https://doi.org/10.1109/jproc.2017.2675998>`__
    dataset is a dataset for remote sensing image scene classification.

    Dataset features:

    * 31,500 images with 0.2-30 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 45 scene classes, 700 images per class

    Dataset format:

    * images are three-channel jpgs

    This dataset uses the train/val/test splits defined in the "In-domain
    representation learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1109/jproc.2017.2675998
    """

    url = "https://hf.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e/NWPU-RESISC45.zip"
    sha256 = "beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53"
    filename = "NWPU-RESISC45.zip"
    directory = "NWPU-RESISC45"

    splits = ("train", "val", "test")
    split_urls = {
        "train": "https://hf.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e/resisc45-train.txt",
        "val": "https://hf.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e/resisc45-val.txt",
        "test": "https://hf.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e/resisc45-test.txt",
    }
    split_sha256s = {
        "train": "ecfa963be4d85eac83665f8be8634abcb4fe6f3546472cc0e87999e2cab4449b",
        "val": "08d81f642526bec240589000af7f49a47e8d071a6e7925b0f36246a78ab64342",
        "test": "e0927e80130b47317a2f18520d98382b6fc56f0d3edd3345140f7d02267c3805",
    }

    def __init__(
        self, root="data", split="train", transforms=None, download=False, checksum=True
    ):
        """Initialize a new RESISC45 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
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
        self.download = download
        self.checksum = checksum
        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, f"resisc45-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())

        def is_in_split(x):
            return os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.directory),
            transforms=transforms,
            is_valid_file=is_in_split,
        )

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
        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                filename=f"resisc45-{split}.txt",
                sha256=self.split_sha256s[split] if self.checksum else None,
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
