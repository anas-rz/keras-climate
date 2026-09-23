"""SKy Images and Photovoltaic Power Dataset (SKIPP'D), ported from
torchgeo.datasets.skippd.
"""

import os

import numpy as np
from einops import rearrange
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive, lazy_import


class SKIPPD(NonGeoDataset):
    """SKy Images and Photovoltaic Power Dataset (SKIPP'D).

    The `SKIPP'D dataset <https://purl.stanford.edu/dj417rh1007>`_
    contains ground-based fish-eye photos of the sky for solar
    forecasting tasks.

    Dataset Format:

    * .hdf5 file containing images and labels

    Dataset Features:

    * fish-eye RGB images (64x64px)
    * power output measurements from 30-kW rooftop PV array
    * 1-min interval across 3 years (2017-2019)

    Nowcast task:

    * 349,372 images under the split key *trainval*
    * 14,003 images under the split key *test*

    Forecast task:

    * 130,412 images under the split key *trainval*
    * 2,462 images under the split key *test*
    * consists of a concatenated RGB time-series of 16 time-steps

    If you use this dataset in your research, please cite:

    * https://doi.org/10.48550/arXiv.2207.00913

    .. note::

       This dataset requires the following additional library to be installed:

       * `<https://pypi.org/project/h5py/>`_ to load the dataset
    """

    url = "https://hf.co/datasets/torchgeo/skippd/resolve/a16c7e200b4618cd93be3143cdb973e3f21498fa/{}"
    sha256 = {
        "forecast": "c9d9695291838ac73e3ee4177dc9cb3fa2178cf132814767436e4df9db11ee8d",
        "nowcast": "5c2e8d0dbece6f50ac299e76c69b6fced437d75cd121a9948ddc12611a6fd603",
    }

    data_file_name = "2017_2019_images_pv_processed_{}.hdf5"
    zipfile_name = "2017_2019_images_pv_processed_{}.zip"

    valid_splits = ("trainval", "test")

    valid_tasks = ("nowcast", "forecast")

    def __init__(
        self,
        root="data",
        split="trainval",
        task="nowcast",
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "trainval", or "test"
            task: one of "nowcast", or "forecast"
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum after downloading files (may be slow)

        Raises:
            AssertionError: if ``task`` or ``split`` is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
            DependencyNotFoundError: If h5py is not installed.
        """
        lazy_import("h5py")

        assert split in self.valid_splits, (
            f"Please choose one of these valid data splits {self.valid_splits}."
        )
        self.split = split

        assert task in self.valid_tasks, (
            f"Please choose one of these valid tasks {self.valid_tasks}."
        )
        self.task = task

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self._verify()

    def __len__(self):
        h5py = lazy_import("h5py")
        with h5py.File(
            os.path.join(self.root, self.data_file_name.format(self.task)), "r"
        ) as f:
            num_datapoints = f[self.split]["pv_log"].shape[0]

        return num_datapoints

    def __getitem__(self, index):
        """Return an index within the dataset."""
        sample = {"image": self._load_image(index)}
        sample.update(self._load_features(index))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index):
        """Load the input image.

        Returns:
            image tensor at index, shape (H, W, C) for nowcast or
            (H, W, T*C) for forecast
        """
        h5py = lazy_import("h5py")
        with h5py.File(
            os.path.join(self.root, self.data_file_name.format(self.task)), "r"
        ) as f:
            arr = f[self.split]["images_log"][index]

        # forecast has dimension [16, 64, 64, 3] but reshape to [64, 64, 48]
        # https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset/blob/main/models/SUNSET_forecast.ipynb
        if self.task == "forecast":
            arr = rearrange(arr, "t h w c -> h w (t c)")

        return ops.cast(ops.convert_to_tensor(np.array(arr)), "float32")

    def _load_features(self, index):
        """Load label."""
        h5py = lazy_import("h5py")
        with h5py.File(
            os.path.join(self.root, self.data_file_name.format(self.task)), "r"
        ) as f:
            label = f[self.split]["pv_log"][index]

        features = {"label": ops.cast(ops.convert_to_tensor(np.array(label)), "float32")}
        return features

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.data_file_name.format(self.task))
        if os.path.exists(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_name.format(self.task))
        if os.path.exists(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self):
        """Download the dataset and extract it."""
        download_url(
            self.url.format(self.zipfile_name.format(self.task)),
            self.root,
            filename=self.zipfile_name.format(self.task),
            sha256=self.sha256[self.task] if self.checksum else None,
        )
        self._extract()

    def _extract(self):
        """Extract the dataset."""
        zipfile_path = os.path.join(self.root, self.zipfile_name.format(self.task))
        extract_archive(zipfile_path, self.root)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        In the ``forecast`` task the latest image is plotted.
        """
        import matplotlib.pyplot as plt

        if self.task == "nowcast":
            image = ops.convert_to_numpy(sample["image"])
            label = float(ops.convert_to_numpy(sample["label"]))
        else:
            image_np = ops.convert_to_numpy(sample["image"]).reshape(64, 64, 16, 3)
            image = image_np[:, :, -1, :]
            label = float(ops.convert_to_numpy(sample["label"])[-1])

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = ops.convert_to_numpy(sample["prediction"])
            prediction = float(
                prediction[-1] if prediction.ndim > 0 else prediction
            )

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image / 255)
        ax.axis("off")

        if show_titles:
            title = f"Label: {label:.3f}"
            if showing_predictions:
                title += f"\nPrediction: {prediction:.3f}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
