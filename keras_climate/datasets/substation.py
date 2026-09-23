"""Substation segmentation dataset (ported from torchgeo.datasets.substation)."""

import glob
import os

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class Substation(NonGeoDataset):
    """Substation dataset.

    The `Substation <https://github.com/Lindsay-Lab/substation-seg>`__
    dataset is curated by TransitionZero and sourced from publicly
    available data repositories, including OpenSreetMap (OSM) and
    Copernicus Sentinel data. The dataset consists of Sentinel-2
    images from 27k+ locations; the task is to segment power-substations,
    which appear in the majority of locations in the dataset. Most
    locations have 4-5 images taken at different timepoints (i.e.,
    revisits).

    Dataset Format:

    * .npz file for each datapoint

    Dataset Features:

    * 26,522 image-mask pairs stored as numpy files.
    * Data from 5 revisits for most locations.
    * Multi-temporal, multi-spectral images (13 channels) paired with masks,
      with a spatial resolution of 228x228 pixels. When
      ``timepoint_aggregation`` is None, images are returned as
      ``T x H x W x C`` tensors.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2409.17363
    """

    all_bands = (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
    )
    rgb_bands = ("B4", "B3", "B2")

    directory = "Substation"
    filename_images = "image_stack.tar.gz"
    filename_masks = "mask.tar.gz"
    url_for_images = "https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/image_stack.tar.gz"
    url_for_masks = "https://storage.googleapis.com/tz-ml-public/substation-over-10km2-csv-main-444e360fd2b6444b9018d509d0e4f36e/mask.tar.gz"
    md5_images = "948706609864d0283f74ee7015f9d032"
    md5_masks = "baa369ececdc2ff80e6ba2b4c7fe147c"

    def __init__(
        self,
        root="data",
        bands=all_bands,
        mask_2d=True,
        num_of_timepoints=4,
        timepoint_aggregation=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize the Substation dataset.

        Args:
            root: path to the directory containing the dataset
            bands: channels to use from the image
            mask_2d: whether to use a 2D (one-hot) mask
            num_of_timepoints: number of timepoints to use for each image
            timepoint_aggregation: one of "concat", "median", "first",
                "random", or None. If None, returns the time-series as
                ``T x H x W x C``.
            transforms: a transform takes input sample and returns a
                transformed version
            download: whether to download the dataset if it is not found
            checksum: whether to verify the dataset after downloading
        """
        self.root = root
        self.bands = bands
        self.mask_2d = mask_2d
        self.num_of_timepoints = num_of_timepoints
        self.timepoint_aggregation = timepoint_aggregation
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.image_dir = os.path.join(root, "image_stack")
        self.mask_dir = os.path.join(root, "mask")
        self._verify()
        self.image_filenames = pd.Series(sorted(os.listdir(self.image_dir)))

    def __getitem__(self, index):
        """Get an item from the dataset by index."""
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, image_filename)

        # (T, C, H, W)
        image = np.load(image_path)["arr_0"]

        # selecting channels
        indices = [self.all_bands.index(band) for band in self.bands]
        image = image[:, indices, :, :]

        # handling multiple images across timepoints
        if image.shape[0] < self.num_of_timepoints:
            # Padding: cycle through existing timepoints
            padded_images = []
            for i in range(self.num_of_timepoints):
                padded_images.append(image[i % image.shape[0]])
            image = np.stack(padded_images)
        elif image.shape[0] > self.num_of_timepoints:
            # Removal: take the most recent timepoints
            image = image[-self.num_of_timepoints :]

        if self.timepoint_aggregation == "concat":
            # (num_of_timepoints*channels, h, w)
            image = np.reshape(image, (-1, image.shape[2], image.shape[3]))
        elif self.timepoint_aggregation == "median":
            image = np.median(image, axis=0)
        elif self.timepoint_aggregation == "first":
            image = image[0]
        elif self.timepoint_aggregation == "random":
            image = image[np.random.randint(image.shape[0])]

        # Move channel axis to the end: (..., C, H, W) -> (..., H, W, C)
        image = np.moveaxis(image, -3, -1)

        mask = np.load(mask_path)["arr_0"]
        mask = np.where(mask != 3, 0, 1).astype("int64")

        if self.mask_2d:
            mask_0 = 1 - mask
            mask = np.stack([mask_0, mask], axis=-1)

        image = ops.convert_to_tensor(image.astype("float32"))
        mask = ops.convert_to_tensor(mask)

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.image_filenames)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        When the image is 4D (``T x H x W x C``), the first two timepoints
        are plotted.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        is_time_series = image.ndim == 4

        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        if is_time_series:
            images = np.clip(np.take(image, rgb_indices, axis=-1) / 4000, 0, 1)
            num_images = min(len(images), 2)
            ncols = num_images + 1
        else:
            image_rgb = np.clip(np.take(image, rgb_indices, axis=-1) / 4000, 0, 1)
            ncols = 2

        mask = ops.convert_to_numpy(sample["mask"])
        if self.mask_2d:
            mask = mask[..., 1]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = ops.convert_to_numpy(sample["prediction"])
            if self.mask_2d:
                prediction = prediction[..., 1]
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))

        if is_time_series:
            for i in range(num_images):
                axs[i].imshow(images[i])
                axs[i].axis("off")
                if show_titles:
                    axs[i].set_title(f"Image {i}")
            axs[num_images].imshow(mask, cmap="gray", interpolation="none")
            axs[num_images].axis("off")
            if show_titles:
                axs[num_images].set_title("Mask")
            if showing_predictions:
                axs[num_images + 1].imshow(prediction, cmap="gray", interpolation="none")
                axs[num_images + 1].axis("off")
                if show_titles:
                    axs[num_images + 1].set_title("Prediction")
        else:
            axs[0].imshow(image_rgb)
            axs[0].axis("off")
            axs[1].imshow(mask, cmap="gray", interpolation="none")
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Image")
                axs[1].set_title("Mask")
            if showing_predictions:
                axs[2].imshow(prediction, cmap="gray", interpolation="none")
                axs[2].axis("off")
                if show_titles:
                    axs[2].set_title("Prediction")

        if suptitle:
            fig.suptitle(suptitle)

        return fig

    def _extract(self):
        """Extract the dataset."""
        img_pathname = os.path.join(self.root, self.filename_images)
        extract_archive(img_pathname)

        mask_pathname = os.path.join(self.root, self.filename_masks)
        extract_archive(mask_pathname)

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        image_path = os.path.join(self.image_dir, "*.npz")
        mask_path = os.path.join(self.mask_dir, "*.npz")
        if glob.glob(image_path) and glob.glob(mask_path):
            return

        # Check if the tar.gz files for images and masks have already been downloaded
        image_exists = os.path.exists(os.path.join(self.root, self.filename_images))
        mask_exists = os.path.exists(os.path.join(self.root, self.filename_masks))
        if image_exists and mask_exists:
            self._extract()
            return

        # If dataset files are missing and download is not allowed, raise an error
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self):
        """Download the dataset and extract it."""
        # Download and verify images
        download_url(
            self.url_for_images,
            self.root,
            filename=self.filename_images,
            md5=self.md5_images if self.checksum else None,
        )
        extract_archive(os.path.join(self.root, self.filename_images), self.root)

        # Download and verify masks
        download_url(
            self.url_for_masks,
            self.root,
            filename=self.filename_masks,
            md5=self.md5_masks if self.checksum else None,
        )
        extract_archive(os.path.join(self.root, self.filename_masks), self.root)
