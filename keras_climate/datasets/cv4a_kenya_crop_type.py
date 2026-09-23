"""CV4A Kenya Crop Type dataset (ported from torchgeo.datasets.cv4a_kenya_crop_type)."""

import os

import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import which


class CV4AKenyaCropType(NonGeoDataset):
    """CV4A Kenya Crop Type Competition dataset.

    The `CV4A Kenya Crop Type Competition
    <https://source.coop/radiantearth/african-crops-kenya-02>`__
    dataset was produced as part of the Crop Type Detection competition at the
    Computer Vision for Agriculture (CV4A) Workshop at the ICLR 2020 conference.
    The objective of the competition was to create a machine learning model to
    classify fields by crop type from images collected during the growing season
    by the Sentinel-2 satellites.

    Consists of 4 tiles of Sentinel 2 imagery from 13 different points in time.

    Each tile has:

    * 13 multi-band observations throughout the growing season. Each observation
      includes 12 bands from Sentinel-2 L2A product, and a cloud probability layer.
      The twelve bands are [B01, B02, B03, B04, B05, B06, B07, B08, B8A,
      B09, B11, B12] (refer to Sentinel-2 documentation for more information about
      the bands). The cloud probability layer is a product of the
      Sentinel-2 atmospheric correction algorithm (Sen2Cor) and provides an estimated
      cloud probability (0-100%) per pixel. All of the bands are mapped to a common
      10 m spatial resolution grid.
    * A raster layer indicating the crop ID for the fields in the training set.
    * A raster layer indicating field IDs for the fields (both training and test sets).
      Fields with a crop ID 0 are the test fields.

    There are 3,286 fields in the train set and 1,402 fields in the test set.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/RDNT.DW605X

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/kenya-crop-challenge"
    tiles = tuple(map(str, range(4)))
    dates = (
        "20190606",
        "20190701",
        "20190706",
        "20190711",
        "20190721",
        "20190805",
        "20190815",
        "20190825",
        "20190909",
        "20190919",
        "20190924",
        "20191004",
        "20191103",
    )
    all_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
        "CLD",
    )
    rgb_bands = ("B04", "B03", "B02")

    # Same for all tiles
    tile_height = 3035
    tile_width = 2016

    def __init__(
        self,
        root="data",
        chip_size=256,
        stride=128,
        bands=all_bands,
        transforms=None,
        download=False,
    ):
        """Initialize a new CV4A Kenya Crop Type Dataset instance.

        Args:
            root: root directory where dataset can be found
            chip_size: size of chips
            stride: spacing between chips, if less than chip_size, then there
                will be overlap between chips
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory

        Raises:
            AssertionError: If *bands* are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert set(bands) <= set(self.all_bands)

        self.root = root
        self.chip_size = chip_size
        self.stride = stride
        self.bands = bands
        self.transforms = transforms
        self.download = download

        self._verify()

        # Calculate the indices that we will use over all tiles
        self.chips_metadata = []
        for tile_index in range(len(self.tiles)):
            for y in [
                *list(range(0, self.tile_height - self.chip_size, stride)),
                self.tile_height - self.chip_size,
            ]:
                for x in [
                    *list(range(0, self.tile_width - self.chip_size, stride)),
                    self.tile_width - self.chip_size,
                ]:
                    self.chips_metadata.append((tile_index, y, x))

    def __getitem__(self, index):
        tile_index, y, x = self.chips_metadata[index]
        tile = self.tiles[tile_index]

        img = self._load_all_image_tiles(tile)
        labels, field_ids = self._load_label_tile(tile)

        # img is [T, H, W, C] (channels-last)
        img = img[:, y : y + self.chip_size, x : x + self.chip_size, :]
        labels = labels[y : y + self.chip_size, x : x + self.chip_size]
        field_ids = field_ids[y : y + self.chip_size, x : x + self.chip_size]

        sample = {
            "image": img,
            "mask": labels,
            "field_ids": field_ids,
            "tile_index": ops.convert_to_tensor(tile_index),
            "x": ops.convert_to_tensor(x),
            "y": ops.convert_to_tensor(y),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.chips_metadata)

    def _load_label_tile(self, tile):
        """Load a single tile of labels and field_ids."""
        directory = os.path.join(self.root, "data", tile)

        with Image.open(os.path.join(directory, f"{tile}_label.tif")) as img:
            array = np.array(img)
            labels = ops.convert_to_tensor(array)

        with Image.open(os.path.join(directory, f"{tile}_field_id.tif")) as img:
            array = np.array(img)
            field_ids = ops.convert_to_tensor(array)

        return labels, field_ids

    def _load_all_image_tiles(self, tile):
        """Load all the imagery (across time) for a single tile.

        Returns imagery of shape [T, H, W, C] (channels-last).
        """
        img = np.zeros(
            (len(self.dates), self.tile_height, self.tile_width, len(self.bands)),
            dtype="float32",
        )

        for date_index, date in enumerate(self.dates):
            img[date_index] = self._load_single_image_tile(tile, date)

        return ops.convert_to_tensor(img)

    def _load_single_image_tile(self, tile, date):
        """Load the imagery for a single tile for a single date.

        Returns an array of shape [H, W, C] (channels-last).
        """
        directory = os.path.join(self.root, "data", tile, date)
        img = np.zeros((self.tile_height, self.tile_width, len(self.bands)), dtype="float32")
        for band_index, band_name in enumerate(self.bands):
            filepath = os.path.join(directory, f"{tile}_{band_name}_{date}.tif")
            with Image.open(filepath) as band_img:
                array = np.array(band_img)
                img[:, :, band_index] = array

        return img

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, "FieldIds.csv")):
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

    def plot(self, sample, show_titles=True, suptitle=None, time_step=0):
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

        if "prediction" in sample:
            prediction = sample["prediction"]
            ncols = 3
        else:
            ncols = 2

        image, mask = sample["image"], sample["mask"]

        image = ops.take(image[time_step], np.array(rgb_indices), axis=-1)

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 10))

        axs[0].imshow(ops.convert_to_numpy(image))
        axs[0].axis("off")
        axs[1].imshow(ops.convert_to_numpy(mask))
        axs[1].axis("off")

        if "prediction" in sample:
            axs[2].imshow(ops.convert_to_numpy(prediction))
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
