"""EuroSAT dataset (ported from torchgeo.datasets.eurosat)."""

import os

import numpy as np
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoClassificationDataset
from .utils import download_and_extract_archive, download_url, extract_archive, rasterio_loader


class EuroSAT(NonGeoClassificationDataset):
    """EuroSAT dataset.

    The `EuroSAT <https://github.com/phelber/EuroSAT>`__ dataset is based on
    Sentinel-2 satellite images covering 13 spectral bands and consists of
    10 target classes with a total of 27,000 labeled and geo-referenced
    images.

    Dataset format:

    * rasters are 13-channel GeoTiffs
    * labels are values in the range [0,9]

    This dataset uses the train/val/test splits defined in the "In-domain
    representation learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite:

    * https://ieeexplore.ieee.org/document/8736785
    * https://ieeexplore.ieee.org/document/8519248
    """

    url = "https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/"
    filename = "EuroSATallBands.zip"
    sha256 = "751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59"

    # For some reason the class directories are actually nested in this directory
    base_dir = os.path.join(
        "ds", "images", "remote_sensing", "otherDatasets", "sentinel_2", "tif"
    )

    splits = ("train", "val", "test")
    split_filenames = {
        "train": "eurosat-train.txt",
        "val": "eurosat-val.txt",
        "test": "eurosat-test.txt",
    }
    split_sha256s = {
        "train": "1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695",
        "val": "b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330",
        "test": "cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592",
    }

    all_band_names = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B8A",
    )

    rgb_bands = ("B04", "B03", "B02")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        root="data",
        split="train",
        bands=BAND_SETS["all"],
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new EuroSAT dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bands: a sequence of band names to load
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
        self.root = root
        self.split = split
        self.tg_transforms = transforms
        self.download = download
        self.checksum = checksum

        assert self.split in {"train", "val", "test"}

        self._validate_bands(bands)
        self.bands = bands
        self.band_indices = np.array(
            [self.all_band_names.index(b) for b in bands if b in self.all_band_names],
            dtype="int64",
        )

        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, self.split_filenames[split])) as f:
            for fn in f:
                valid_fns.add(fn.strip().replace(".jpg", ".tif"))

        def is_in_split(x):
            return os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transforms,
            loader=rasterio_loader,
            is_valid_file=is_in_split,
        )

    def __getitem__(self, index):
        image, label = self._load_image(index)

        image = ops.take(image, self.band_indices, axis=-1)
        image = ops.cast(image, "float32")
        sample = {"image": image, "label": label}

        if self.tg_transforms is not None:
            sample = self.tg_transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        filename = os.path.join(self.root, self.split_filenames[self.split])
        if not os.path.isfile(filename):
            if self.download:
                download_url(
                    self.url + self.split_filenames[self.split],
                    self.root,
                    sha256=self.split_sha256s[self.split] if self.checksum else None,
                )
            else:
                raise DatasetNotFoundError(self)

        directory = os.path.join(self.root, self.base_dir)
        zipfile = os.path.join(self.root, self.filename)
        if os.path.isdir(directory):
            return
        elif os.path.isfile(zipfile):
            extract_archive(zipfile)
        elif self.download:
            download_and_extract_archive(
                self.url + self.filename,
                self.root,
                sha256=self.sha256 if self.checksum else None,
            )
        else:
            raise DatasetNotFoundError(self)

    def _validate_bands(self, bands):
        """Validate list of bands.

        Raises:
            ValueError: if an invalid band name is provided
        """
        for band in bands:
            if band not in self.all_band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        import matplotlib.pyplot as plt

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = np.take(ops.convert_to_numpy(sample["image"]), rgb_indices, axis=-1)
        image = np.clip(image / 3000, 0, 1)

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


class EuroSATSpatial(EuroSAT):
    """Overrides the default EuroSAT dataset splits.

    Splits the data into training, validation, and test sets based on
    longitude. The splits are distributed as 60%, 20%, and 20% respectively.
    """

    split_filenames = {
        "train": "eurosat-spatial-train.txt",
        "val": "eurosat-spatial-val.txt",
        "test": "eurosat-spatial-test.txt",
    }
    split_sha256s = {
        "train": "2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f",
        "val": "6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b",
        "test": "de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452",
    }


class EuroSAT100(EuroSAT):
    """Subset of EuroSAT containing only 100 images.

    Intended for tutorials and demonstrations, not for benchmarking.
    Maintains the same file structure, classes, and train-val-test split.
    Each class has 10 images (6 train, 2 val, 2 test), for a total of 100
    images.
    """

    filename = "EuroSAT100.zip"
    sha256 = "2ed4bb4a6808004c98691f64b366827f7783c76a49151e7c2b70423eb77a5b76"

    split_filenames = {
        "train": "eurosat-100-train.txt",
        "val": "eurosat-100-val.txt",
        "test": "eurosat-100-test.txt",
    }
    split_sha256s = {
        "train": "7f2416377fc379d43197512ef2f7582e87f62c7b2b63ac576bd5c18f110e0a20",
        "val": "09e4744d15299c0e5b39a5bb211ce245b8dbaeafc5353ce3372f5db69f6e59d8",
        "test": "576738254dffa4fd3cdefed1763ecbd9f6e2b4de0f5ec8a9aa4ff301a6d8e00e",
    }
