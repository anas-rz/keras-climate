"""SustainBench Crop Yield dataset (ported from torchgeo.datasets.sustainbench_crop_yield)."""

import os

import numpy as np
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class SustainBenchCropYield(NonGeoDataset):
    """SustainBench Crop Yield Dataset.

    This dataset contains MODIS band histograms and soybean yield
    estimates for selected counties in the USA, Argentina and Brazil.
    The dataset is part of the
    `SustainBench <https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_yield.html>`_
    datasets for tackling the UN Sustainable Development Goals (SDGs).

    Dataset Format:

    * .npz files of stacked samples

    Dataset Features:

    * input histogram of 7 surface reflectance and 2 surface temperature
      bands from MODIS pixel values in 32 ranges across 32 timesteps
      resulting in 32x32x9 input images
    * regression target value of soybean yield in metric tonnes per
      harvested hectare

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1145/3209811.3212707
    * https://doi.org/10.1609/aaai.v31i1.11172
    """

    valid_countries = ("usa", "brazil", "argentina")

    sha256 = "ff66f83a91a16b302c731c7efa4020ecf323d96beca2f1f9196a643efdb8ea4a"

    url = "https://hf.co/datasets/torchgeo/sustainbench_crop_yield/resolve/eceefda0b866c321c18baa256205d21fa6f5eb8c/soybeans_updated.zip"

    dir = "soybeans"

    valid_splits = ("train", "dev", "test")

    def __init__(
        self,
        root="data",
        split="train",
        countries=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "dev", or "test"
            countries: which countries to include in the dataset
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum after downloading files
                (may be slow)

        Raises:
            AssertionError: if ``countries`` contains invalid countries or
                if ``split`` is invalid
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        if countries is None:
            countries = ["usa"]
        assert set(countries).issubset(self.valid_countries), (
            f"Please choose a subset of these valid countried: {self.valid_countries}."
        )
        self.countries = countries

        assert split in self.valid_splits, (
            f"Pleas choose one of these valid data splits {self.valid_splits}."
        )
        self.split = split

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.images = []
        self.features = []

        for country in self.countries:
            image_file_path = os.path.join(
                self.root, self.dir, country, f"{self.split}_hists.npz"
            )
            target_file_path = image_file_path.replace("_hists", "_yields")
            years_file_path = image_file_path.replace("_hists", "_years")
            ndvi_file_path = image_file_path.replace("_hists", "_ndvi")

            npz_file = np.load(image_file_path)["data"]
            target_npz_file = np.load(target_file_path)["data"]
            year_npz_file = np.load(years_file_path)["data"]
            ndvi_npz_file = np.load(ndvi_file_path)["data"]
            num_data_points = npz_file.shape[0]
            for idx in range(num_data_points):
                sample = npz_file[idx].astype("float32")
                self.images.append(sample)

                target = target_npz_file[idx]
                year = year_npz_file[idx]
                ndvi = ndvi_npz_file[idx]

                features = {
                    "label": np.array(target, dtype="float32"),
                    "year": np.array(int(year), dtype="int64"),
                    "ndvi": ndvi.astype("float32"),
                }
                self.features.append(features)

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.images)

    def __getitem__(self, index):
        """Return an index within the dataset."""
        sample = {"image": ops.convert_to_tensor(self.images[index])}
        for key, value in self.features[index].items():
            sample[key] = ops.convert_to_tensor(value)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.dir)
        if os.path.exists(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.dir) + ".zip"
        if os.path.exists(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self):
        """Download the dataset and extract it."""
        download_url(
            self.url,
            self.root,
            filename=self.dir + ".zip",
            sha256=self.sha256 if self.checksum else None,
        )
        self._extract()

    def _extract(self):
        """Extract the dataset."""
        zipfile_path = os.path.join(self.root, self.dir) + ".zip"
        extract_archive(zipfile_path, self.root)

    def plot(self, sample, show_titles=True, suptitle=None, band_idx=0):
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure
            band_idx: which of the nine histograms to index
        """
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        label = float(ops.convert_to_numpy(sample["label"]))

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = float(ops.convert_to_numpy(sample["prediction"]))

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image[:, :, band_idx])
        ax.axis("off")

        if show_titles:
            title = f"Label: {label:.3f}"
            if showing_predictions:
                title += f"\nPrediction: {prediction:.3f}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
