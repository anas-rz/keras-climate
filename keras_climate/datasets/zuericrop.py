"""ZueriCrop dataset (ported from torchgeo.datasets.zuericrop)."""

import os

import numpy as np
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import download_url, lazy_import, quantile_normalization


class ZueriCrop(NonGeoDataset):
    """ZueriCrop dataset.

    The `ZueriCrop <https://github.com/0zgur0/multi-stage-convSTAR-network>`__
    dataset is a dataset for time-series instance segmentation of crops.

    Dataset features:

    * Sentinel-2 multispectral imagery
    * instance masks of 48 crop categories
    * nine multispectral bands
    * 116k images with 10 m per pixel resolution (24x24 px)
    * ~28k time-series containing 142 images each

    Dataset format:

    * single hdf5 dataset containing images, semantic masks, and instance
      masks
    * data is parsed into images and instance masks, boxes, and labels
    * one mask per time-series

    Dataset classes:

    * 48 fine-grained hierarchical crop `categories
      <https://github.com/0zgur0/multi-stage-convSTAR-network/blob/fa92b5b3cb77f5171c5c3be740cd6e6395cc29b6/labels.csv>`_

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1016/j.rse.2021.112603

    .. note::

       This dataset requires the following additional library to be
       installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset
    """

    url = "https://hf.co/datasets/isaaccorley/zuericrop/resolve/8ac0f416fbaab032d8670cc55f984b9f079e86b2/"
    sha256s = (
        "738536d3a28154e9a47ab8431b325a0213e80e7b75e8df9f7f42cc815b322fde",
        "9bd634d09075e5a196a6b32ed81cb75dba50aa926644f1deb2a0a27f27af82ad",
    )
    filenames = ("ZueriCrop.hdf5", "labels.csv")

    band_names = ("NIR", "B03", "B02", "B04", "B05", "B06", "B07", "B11", "B12")
    rgb_bands = ("B04", "B03", "B02")

    def __init__(self, root="data", bands=band_names, transforms=None, download=False, checksum=True):
        """Initialize a new ZueriCrop dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
            DependencyNotFoundError: If h5py is not installed.
        """
        lazy_import("h5py")

        self._validate_bands(bands)
        self.band_indices = np.array([self.band_names.index(b) for b in bands], dtype="int64")

        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.filepath = os.path.join(root, "ZueriCrop.hdf5")

        self._verify()

    def __getitem__(self, index):
        """Return a sample containing image, mask, bounding boxes, and target label."""
        image = self._load_image(index)
        mask, boxes, label = self._load_target(index)

        sample = {"image": image, "mask": mask, "bbox_xyxy": boxes, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        h5py = lazy_import("h5py")
        with h5py.File(self.filepath, "r") as f:
            length = f["data"].shape[0]
        return length

    def _load_image(self, index):
        """Load a single image, returned as a [T, H, W, C] tensor."""
        h5py = lazy_import("h5py")
        with h5py.File(self.filepath, "r") as f:
            array = f["data"][index, ...]

        tensor = ops.convert_to_tensor(array.astype("float32"))
        tensor = ops.take(tensor, self.band_indices, axis=-1)
        return tensor

    def _load_target(self, index):
        """Load the target mask for a single image."""
        h5py = lazy_import("h5py")
        with h5py.File(self.filepath, "r") as f:
            mask_array = f["gt"][index, ...]
            instance_array = f["gt_instance"][index, ...]

        # gt/gt_instance store single-channel labels with shape (H, W, 1)
        mask_array = mask_array[..., 0]
        instance_array = instance_array[..., 0]

        # Convert instance mask of N instances to N binary instance masks
        instance_ids = np.unique(instance_array)
        # Exclude a mask for unknown/background
        instance_ids = instance_ids[instance_ids != 0]

        masks = instance_array[None, :, :] == instance_ids[:, None, None]

        # Parse labels for each instance
        labels_list = []
        for mask in masks:
            label = mask_array[mask]
            label = np.unique(label)[0]
            labels_list.append(label)

        # Get bounding boxes for each instance
        boxes_list = []
        for mask in masks:
            pos = np.where(mask)
            xmin = pos[1].min()
            xmax = pos[1].max()
            ymin = pos[0].min()
            ymax = pos[0].max()
            boxes_list.append([xmin, ymin, xmax, ymax])

        masks = masks.astype("uint8")
        boxes = np.array(boxes_list, dtype="float32").reshape(-1, 4)
        labels = np.array(labels_list, dtype="int64").reshape(-1)

        return (
            ops.convert_to_tensor(masks),
            ops.convert_to_tensor(boxes),
            ops.convert_to_tensor(labels),
        )

    def _verify(self):
        """Verify the integrity of the dataset."""
        exists = []
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            exists.append(os.path.exists(filepath))

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the dataset."""
        for filename, sha256 in zip(self.filenames, self.sha256s):
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                download_url(
                    self.url + filename, self.root, filename=filename, sha256=sha256 if self.checksum else None
                )

    def _validate_bands(self, bands):
        """Validate list of bands.

        Raises:
            ValueError: if an invalid band name is provided
        """
        for band in bands:
            if band not in self.band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

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

        ncols = 2
        image = ops.convert_to_numpy(sample["image"])[time_step]
        image = np.take(image, rgb_indices, axis=-1)
        image = ops.convert_to_numpy(quantile_normalization(image))
        image = np.clip(image * 255, 0, 255).astype("uint8")

        mask = ops.convert_to_numpy(ops.argmax(sample["mask"], axis=0))

        if "prediction" in sample:
            ncols += 1
            preds = ops.convert_to_numpy(ops.argmax(sample["prediction"], axis=0))

        fig, axs = plt.subplots(ncols=ncols, figsize=(10 * ncols, 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if "prediction" in sample:
            axs[2].imshow(preds)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
