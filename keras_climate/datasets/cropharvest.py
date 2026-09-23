"""CropHarvest dataset (ported from torchgeo.datasets.cropharvest)."""

import glob
import json
import os

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive, lazy_import


class CropHarvest(NonGeoDataset):
    """CropHarvest dataset.

    `CropHarvest <https://github.com/nasaharvest/cropharvest>`__ is a
    crop classification dataset.

    Dataset features:

    * single pixel time series with crop-type labels
    * 18 bands per image over 12 months

    Dataset format:

    * arrays are 12x18 with 18 bands over 12 months

    Dataset properties:

    1. is_crop - whether or not a single pixel contains cropland
    2. classification_label - optional field identifying a specific crop type
    3. dataset - source dataset for the imagery
    4. lat - latitude
    5. lon - longitude

    If you use this dataset in your research, please cite the following paper:

    * https://neurips.cc/virtual/2021/29874

    This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset
    """

    # https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/bands.py
    all_bands = (
        "VV",
        "VH",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
        "temperature_2m",
        "total_precipitation",
        "elevation",
        "slope",
        "NDVI",
    )
    rgb_bands = ("B4", "B3", "B2")

    features_url = "https://zenodo.org/records/7257688/files/features.tar.gz?download=1"
    labels_url = "https://zenodo.org/records/7257688/files/labels.geojson?download=1"
    file_dict = {
        "features": {
            "url": features_url,
            "filename": "features.tar.gz",
            "extracted_filename": os.path.join("features", "arrays"),
            "md5": "cad4df655c75caac805a80435e46ee3e",
        },
        "labels": {
            "url": labels_url,
            "filename": "labels.geojson",
            "extracted_filename": "labels.geojson",
            "md5": "bf7bae6812fc7213481aff6a2e34517d",
        },
    }

    def __init__(self, root="data", transforms=None, download=False, checksum=True):
        """Initialize a new CropHarvest dataset instance.

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
            DependencyNotFoundError: If h5py is not installed.
        """
        lazy_import("h5py")

        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        self._verify()

        self.files = self._load_features(self.root)
        self.labels = self._load_labels(self.root)
        classes = self.labels["properties.label"].unique()
        classes = classes[classes != np.array(None)]
        self.classes = np.insert(classes, 0, ["None", "Other"])

    def __getitem__(self, index):
        files = self.files[index]
        data = self._load_array(files["chip"])

        label = self._load_label(files["index"], files["dataset"])
        sample = {"array": data, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _load_features(self, root):
        """Return the paths of the files in the dataset."""
        files = []
        chips = glob.glob(
            os.path.join(root, self.file_dict["features"]["extracted_filename"], "*.h5")
        )
        chips = sorted(os.path.basename(chip) for chip in chips)
        for chip in chips:
            chip_path = os.path.join(
                root, self.file_dict["features"]["extracted_filename"], chip
            )
            index = chip.split("_")[0]
            dataset = chip.split("_")[1][:-3]
            files.append({"chip": chip_path, "index": index, "dataset": dataset})
        return files

    def _load_labels(self, root):
        """Return the labels for each feature as a dataframe."""
        filename = self.file_dict["labels"]["extracted_filename"]
        with open(os.path.join(root, filename), encoding="utf8") as f:
            data = json.load(f)
            df = pd.json_normalize(data["features"])
            return df

    def _load_array(self, path):
        """Load an individual single pixel time series."""
        h5py = lazy_import("h5py")
        filename = os.path.join(path)
        with h5py.File(filename, "r") as f:
            array = f.get("array")[()]
            return ops.convert_to_tensor(array)

    def _load_label(self, idx, dataset):
        """Load the crop-type label for a single pixel time series."""
        index = int(idx)
        row = self.labels[
            (self.labels["properties.index"] == index)
            & (self.labels["properties.dataset"] == dataset)
        ]
        properties = row.to_dict(orient="records")[0]
        label = "None"
        if isinstance(properties["properties.label"], str):
            label = properties["properties.label"]
        elif properties["properties.is_crop"] == 1:
            label = "Other"

        return ops.convert_to_tensor(np.where(self.classes == label)[0][0])

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if feature files already exist
        feature_path = os.path.join(
            self.root, self.file_dict["features"]["extracted_filename"]
        )
        feature_path_zip = os.path.join(self.root, self.file_dict["features"]["filename"])
        label_path = os.path.join(self.root, self.file_dict["labels"]["extracted_filename"])
        # Check if labels exist
        if os.path.exists(label_path):
            # Check if features exist
            if os.path.exists(feature_path):
                return
            # Check if features are downloaded in zip format
            if os.path.exists(feature_path_zip):
                self._extract()
                return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self):
        """Download the dataset and extract it."""
        features_path = os.path.join(self.file_dict["features"]["filename"])
        download_url(
            self.file_dict["features"]["url"],
            self.root,
            filename=features_path,
            md5=self.file_dict["features"]["md5"] if self.checksum else None,
        )

        download_url(
            self.file_dict["labels"]["url"],
            self.root,
            filename=os.path.join(self.file_dict["labels"]["filename"]),
            md5=self.file_dict["labels"]["md5"] if self.checksum else None,
        )

    def _extract(self):
        """Extract the dataset."""
        features_path = os.path.join(self.root, self.file_dict["features"]["filename"])
        extract_archive(features_path)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset using bands for Agriculture RGB composite."""
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots()
        bands = [self.all_bands.index(band) for band in self.rgb_bands]
        rgb = ops.convert_to_numpy(sample["array"])[:, bands] / 3000
        axs.imshow(rgb[None, ...])
        if show_titles:
            label = int(ops.convert_to_numpy(sample["label"]))
            axs.set_title(f"Crop type: {self.classes[label]}")
        axs.set_xticks(np.arange(12))
        axs.set_xticklabels(np.arange(12) + 1)
        axs.set_yticks([])
        axs.set_xlabel("Month")
        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
