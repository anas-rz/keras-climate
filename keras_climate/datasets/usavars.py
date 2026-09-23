"""USAVars dataset (ported from torchgeo.datasets.usavars)."""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import rasterio
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class USAVars(NonGeoDataset):
    """USAVars dataset.

    The USAVars dataset is reproduction of the dataset used in the paper "`A
    generalizable and accessible approach to machine learning with global
    satellite imagery <https://doi.org/10.1038/s41467-021-24638-z>`_".
    Specifically, this dataset includes 1 sq km. crops of NAIP imagery
    resampled to 4m/px cenetered on ~100k points that are sampled randomly
    from the contiguous states in the USA. Each point contains three
    continuous valued labels (taken from the dataset released in the
    paper): tree cover percentage, elevation, and population density.

    Dataset format:

    * images are 4-channel GeoTIFFs
    * labels are singular float values

    Dataset labels:

    * tree cover
    * elevation
    * population density

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1038/s41467-021-24638-z
    """

    data_url = "https://hf.co/datasets/torchgeo/usavars/resolve/01377abfaf50c0cc8548aaafb79533666bbf288f/{}"
    dirname = "uar"

    sha256 = "a202ed19a720bccc0cb313bb43145ef5b546a6c2a5cbf54955e1e9506a5f3b0b"

    label_urls = {
        "housing": data_url.format("housing.csv"),
        "income": data_url.format("income.csv"),
        "roads": data_url.format("roads.csv"),
        "nightlights": data_url.format("nightlights.csv"),
        "population": data_url.format("population.csv"),
        "elevation": data_url.format("elevation.csv"),
        "treecover": data_url.format("treecover.csv"),
    }

    split_metadata = {
        "train": {
            "url": data_url.format("train_split.txt"),
            "filename": "train_split.txt",
            "sha256": "d1a7a3be2b9328e03affd62d23347eb7e76e786079c012107ad65cb652d97d93",
        },
        "val": {
            "url": data_url.format("val_split.txt"),
            "filename": "val_split.txt",
            "sha256": "edec887529ce83b3ae1fb9b549621b6d16df356fb9e83a4134f073761d147660",
        },
        "test": {
            "url": data_url.format("test_split.txt"),
            "filename": "test_split.txt",
            "sha256": "364f7ae1de7803a026e41a8624b540bfab756dcc2fe87a04f036b4def27d93fb",
        },
    }

    ALL_LABELS = ("treecover", "elevation", "population")

    def __init__(
        self,
        root="data",
        split="train",
        labels=ALL_LABELS,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new USAVars dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            labels: list of labels to include
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: if invalid labels are provided
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.root = root

        assert split in self.split_metadata
        self.split = split

        for lab in labels:
            assert lab in self.ALL_LABELS

        self.labels = labels
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

        self.label_dfs = {
            lab: pd.read_csv(os.path.join(self.root, lab + ".csv"), index_col="ID")
            for lab in self.labels
        }

    def __getitem__(self, index):
        """Return an index within the dataset."""
        tif_file = self.files[index]
        id_ = tif_file[5:-4]

        sample = {
            "labels": ops.convert_to_tensor(
                np.array(
                    [self.label_dfs[lab].loc[id_][lab] for lab in self.labels],
                    dtype="float32",
                )
            ),
            "image": self._load_image(os.path.join(self.root, "uar", tif_file)),
            "centroid_lat": ops.convert_to_tensor(
                np.array([self.label_dfs[self.labels[0]].loc[id_]["lat"]], dtype="float32")
            ),
            "centroid_lon": ops.convert_to_tensor(
                np.array([self.label_dfs[self.labels[0]].loc[id_]["lon"]], dtype="float32")
            ),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def _load_files(self):
        """Loads file names."""
        with open(os.path.join(self.root, f"{self.split}_split.txt")) as f:
            files = f.read().splitlines()
        return files

    def _load_image(self, path):
        """Load a single image."""
        with rasterio.open(path) as f:
            array = f.read().astype("float32")
            # (C, H, W) -> (H, W, C)
            array = np.transpose(array, (1, 2, 0))
            return ops.convert_to_tensor(array)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "uar")
        csv_pathname = os.path.join(self.root, "*.csv")
        split_pathname = os.path.join(self.root, "*_split.txt")

        csv_split_count = (len(glob.glob(csv_pathname)), len(glob.glob(split_pathname)))
        if glob.glob(pathname) and csv_split_count == (7, 3):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.dirname + ".zip")
        if glob.glob(pathname) and csv_split_count == (7, 3):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self):
        """Download the dataset."""
        for f_name in self.label_urls:
            download_url(self.label_urls[f_name], self.root, filename=f_name + ".csv")

        download_url(
            self.data_url.format(self.dirname + ".zip"),
            self.root,
            sha256=self.sha256 if self.checksum else None,
        )

        for metadata in self.split_metadata.values():
            download_url(
                metadata["url"],
                self.root,
                sha256=metadata["sha256"] if self.checksum else None,
            )

    def _extract(self):
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.dirname + ".zip"))

    def plot(self, sample, show_titles=True, suptitle=None, show_labels=None):
        """Plot a sample from the dataset.

        .. versionchanged::
            Renamed show_labels to show_titles - keep show_labels as
            deprecated alias
        """
        import matplotlib.pyplot as plt

        if show_labels is not None:
            warnings.warn(
                "The show_labels parameter is deprecated, use show_titles instead.",
                DeprecationWarning,
            )
            show_titles = show_labels

        image = ops.convert_to_numpy(sample["image"])[:, :, :3]  # get RGB inds

        fig, axs = plt.subplots(figsize=(10, 10))
        axs.imshow(image)
        axs.axis("off")

        if show_titles:
            labels = [(lab, val) for lab, val in sample.items() if lab != "image"]
            label_string = ""
            for lab, val in labels:
                val_np = ops.convert_to_numpy(val)
                label_string += f"{lab}={round(float(val_np.reshape(-1)[0]), 2)} "
            axs.set_title(label_string)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
