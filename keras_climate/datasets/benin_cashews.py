"""Smallholder Cashew Plantations in Benin dataset (ported from
torchgeo.datasets.benin_cashews).
"""

import json
import os

import numpy as np
import rasterio
import rasterio.features
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import which


class BeninSmallHolderCashews(NonGeoDataset):
    r"""Smallholder Cashew Plantations in Benin dataset.

    This dataset contains labels for cashew plantations in a 120 km\
    :sup:`2` area in the center of Benin. Each pixel is classified for
    Well-managed plantation, Poorly-managed plantation, No plantation and
    other classes. The labels are generated using a combination of ground
    data collection with a handheld GPS device, and final corrections based
    on Airbus Pléiades imagery. See `this website
    <https://source.coop/technoserve/cashews-benin>`__ for dataset details.

    Specifically, the data consists of Sentinel 2 imagery from a 120 km\
    :sup:`2` area in the center of Benin over 71 points in time from
    11/05/2019 to 10/30/2020 and polygon labels for 6 classes:

    0. No data
    1. Well-managed plantation
    2. Poorly-managed planatation
    3. Non-plantation
    4. Residential
    5. Background
    6. Uncertain

    If you use this dataset in your research, please cite the following:

    * https://source.coop/technoserve/cashews-benin/

    .. note::

       This dataset requires the following additional library to be
       installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to
         download the dataset from Source Cooperative.
    """

    url = "https://radiantearth.blob.core.windows.net/mlhub/technoserve-cashew-benin"
    dates = (
        "20191105",
        "20191110",
        "20191115",
        "20191120",
        "20191130",
        "20191205",
        "20191210",
        "20191215",
        "20191220",
        "20191225",
        "20191230",
        "20200104",
        "20200109",
        "20200114",
        "20200119",
        "20200124",
        "20200129",
        "20200208",
        "20200213",
        "20200218",
        "20200223",
        "20200228",
        "20200304",
        "20200309",
        "20200314",
        "20200319",
        "20200324",
        "20200329",
        "20200403",
        "20200408",
        "20200413",
        "20200418",
        "20200423",
        "20200428",
        "20200503",
        "20200508",
        "20200513",
        "20200518",
        "20200523",
        "20200528",
        "20200602",
        "20200607",
        "20200612",
        "20200617",
        "20200622",
        "20200627",
        "20200702",
        "20200707",
        "20200712",
        "20200717",
        "20200722",
        "20200727",
        "20200801",
        "20200806",
        "20200811",
        "20200816",
        "20200821",
        "20200826",
        "20200831",
        "20200905",
        "20200910",
        "20200915",
        "20200920",
        "20200925",
        "20200930",
        "20201010",
        "20201015",
        "20201020",
        "20201025",
        "20201030",
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

    classes = (
        "No data",
        "Well-managed planatation",
        "Poorly-managed planatation",
        "Non-planatation",
        "Residential",
        "Background",
        "Uncertain",
    )

    # Same for all tiles
    tile_height = 1186
    tile_width = 1122

    def __init__(
        self,
        root="data",
        chip_size=256,
        stride=128,
        bands=all_bands,
        transforms=None,
        download=False,
    ):
        """Initialize a new Benin Smallholder Cashew Plantations Dataset
        instance.

        Args:
            root: root directory where dataset can be found
            chip_size: size of chips
            stride: spacing between chips, if less than chip_size, then
                there will be overlap between chips
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory

        Raises:
            AssertionError: If *bands* is invalid.
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
        for y in [
            *list(range(0, self.tile_height - self.chip_size, stride)),
            self.tile_height - self.chip_size,
        ]:
            for x in [
                *list(range(0, self.tile_width - self.chip_size, stride)),
                self.tile_width - self.chip_size,
            ]:
                self.chips_metadata.append((y, x))

    def __getitem__(self, index):
        y, x = self.chips_metadata[index]

        img, transform = self._load_all_imagery()
        labels = self._load_mask(transform)

        img = img[:, y : y + self.chip_size, x : x + self.chip_size, :]
        labels = labels[y : y + self.chip_size, x : x + self.chip_size]

        sample = {
            "image": img,
            "mask": labels,
            "x": ops.convert_to_tensor(np.array(x)),
            "y": ops.convert_to_tensor(np.array(y)),
            "transform": ops.convert_to_tensor(np.array(list(transform))),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.chips_metadata)

    def _load_all_imagery(self):
        """Load all the imagery (across time) for the dataset.

        Returns:
            imagery of shape (70, 1186, 1122, number of bands) where 70 is
            the number of points in time, 1186 is the tile height, and 1122
            is the tile width; and the rasterio affine transform, mapping
            pixel coordinates to geo coordinates.
        """
        scenes = []
        transform = None
        for date in self.dates:
            scene, transform = self._load_single_scene(date)
            scenes.append(scene)

        img = ops.convert_to_tensor(np.stack(scenes, axis=0).astype("float32"))
        return img, transform

    def _load_single_scene(self, date):
        """Load the imagery for a single date.

        Returns:
            a (H, W, num_bands) array, and the rasterio affine transform
            mapping pixel coordinates to geo coordinates.
        """
        bands_arr = []
        transform = None
        for band_name in self.bands:
            filepath = os.path.join(
                self.root,
                "imagery",
                "00",
                f"00_{date}",
                f"00_{date}_{band_name}_10m.tif",
            )
            with rasterio.open(filepath) as src:
                transform = src.transform  # same transform for every band
                array = src.read(1).astype(np.float32)
                bands_arr.append(array)

        # (H, W, C)
        img = np.stack(bands_arr, axis=-1)
        return img, transform

    def _load_mask(self, transform):
        """Rasterizes the dataset's labels (in geojson format)."""
        with open(os.path.join(self.root, "labels", "00.geojson")) as f:
            geojson = json.load(f)

        labels = [
            (feature["geometry"], feature["properties"]["class"])
            for feature in geojson["features"]
        ]

        mask_data = rasterio.features.rasterize(
            labels,
            out_shape=(self.tile_height, self.tile_width),
            fill=0,  # nodata value
            transform=transform,
            all_touched=False,
            dtype=np.uint8,
        )

        mask = ops.cast(ops.convert_to_tensor(mask_data), "int64")
        return mask

    def _verify(self):
        """Verify the integrity of the dataset."""
        if os.path.exists(os.path.join(self.root, "labels", "00.geojson")):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

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

        image = ops.convert_to_numpy(sample["image"])[time_step]
        image = np.take(image, rgb_indices, axis=-1)
        image = np.clip(image / 3000, 0, 1)
        mask = ops.convert_to_numpy(sample["mask"])

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = ops.convert_to_numpy(sample["prediction"])
            num_panels += 1

        fig, axs = plt.subplots(ncols=num_panels, figsize=(4 * num_panels, 4))

        axs[0].imshow(image)
        axs[0].axis("off")
        if show_titles:
            axs[0].set_title(f"t={time_step}")

        axs[1].imshow(mask, vmin=0, vmax=6, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions, vmin=0, vmax=6, interpolation="none")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
