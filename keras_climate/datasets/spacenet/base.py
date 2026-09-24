"""SpaceNet abstract base class (ported from torchgeo.datasets.spacenet.base)."""

import glob
import json
import os
import re
from abc import ABC, abstractmethod
from json.decoder import JSONDecodeError

import geopandas as gpd
import numpy as np
import rasterio as rio
from keras import ops
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize

from ..errors import DatasetNotFoundError
from ..geo import NonGeoDataset
from ..utils import check_integrity, extract_archive, quantile_normalization, which


class SpaceNet(NonGeoDataset, ABC):
    """Abstract base class for the SpaceNet datasets.

    The `SpaceNet <https://spacenet.ai/datasets/>`__ datasets are a set of
    datasets that all together contain >11M building footprints and ~20,000 km
    of road labels mapped over high-resolution satellite imagery obtained from
    a variety of sensors such as Worldview-2, Worldview-3 and Dove.

    .. note::

       The SpaceNet datasets require the following additional library to be
       installed: `AWS CLI <https://aws.amazon.com/cli/>`_, to download the
       dataset from AWS.
    """

    url = "s3://spacenet-dataset/spacenet/{dataset_id}/tarballs/{tarball}"
    directory_glob = os.path.join("**", "AOI_{aoi}_*", "{product}")
    image_glob = "*.tif"
    mask_glob = "*.geojson"
    file_regex = r"_img(\d+)\."
    chip_size = {}

    cities = {
        1: "Rio",
        2: "Vegas",
        3: "Paris",
        4: "Shanghai",
        5: "Khartoum",
        6: "Atlanta",
        7: "Moscow",
        8: "Mumbai",
        9: "San Juan",
        10: "Dar Es Salaam",
        11: "Rotterdam",
    }

    @property
    @abstractmethod
    def dataset_id(self):
        """Dataset ID."""

    @property
    @abstractmethod
    def tarballs(self):
        """Mapping of `tarballs[split][aoi]` to a list of tarball names."""

    @property
    @abstractmethod
    def md5s(self):
        """Mapping of `md5s[split][aoi]` to a list of md5 checksums."""

    @property
    @abstractmethod
    def valid_aois(self):
        """Mapping of valid_aois[split] = [aois]."""

    @property
    @abstractmethod
    def valid_images(self):
        """Mapping of valid_images[split] = [images]."""

    @property
    @abstractmethod
    def valid_masks(self):
        """List of valid masks."""

    def __init__(
        self,
        root="data",
        split="train",
        aois=None,
        image=None,
        mask=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new SpaceNet Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: 'train' or 'test' split
            aois: areas of interest
            image: image selection
            mask: mask selection
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version.
            download: if True, download dataset and store it in the root
                directory.
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: If any invalid arguments are passed.
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        if aois is None:
            aois = []
        self.root = root
        self.split = split
        self.aois = aois or self.valid_aois[split]
        self.image = image or self.valid_images[split][0]
        self.mask = mask or self.valid_masks[0]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        assert self.split in {"train", "test"}
        assert set(self.aois) <= set(self.valid_aois[split])
        assert self.image in self.valid_images[split]
        assert self.mask in self.valid_masks

        self._verify()

        if self.split == "train":
            assert len(self.images) == len(self.masks)

    def __len__(self):
        return len(self.images)

    def _load_image(self, path):
        """Load a single image.

        Returns:
            (image, transform, crs). ``image`` is channels-last (``H x W x C``).
        """
        with rio.open(path) as img:
            out_shape = (img.count, img.height, img.width)
            if self.image in self.chip_size:
                out_shape = (img.count, *self.chip_size[self.image])
            array = img.read(out_shape=out_shape, resampling=Resampling.bilinear)
            array = np.transpose(array.astype(np.float32), (1, 2, 0))
            tensor = ops.convert_to_tensor(array)
            # https://pyproj4.github.io/pyproj/stable/crs_compatibility.html#rasterio
            with rio.Env(OSR_WKT_FORMAT="WKT2_2018"):
                crs = CRS.from_user_input(img.crs)
            return tensor, img.transform, crs

    def _load_mask(self, path, tfm, raster_crs, shape):
        """Rasterizes the dataset's labels (in geojson format)."""
        try:
            # Raises a JSONDecodeError for empty files
            with open(path) as f:
                json.load(f)

            gdf = gpd.read_file(path)

        except JSONDecodeError:
            gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

        if gdf.empty:
            labels = []
        else:
            gdf.to_crs(raster_crs, inplace=True)
            gdf.dropna(subset=["geometry"], inplace=True)
            labels = gdf.geometry.tolist()

        if labels:
            mask = rasterize(
                labels,
                out_shape=shape,
                fill=0,
                transform=tfm,
                all_touched=False,
                dtype=np.int64,
            )
        else:
            mask = np.zeros(shape=shape, dtype=np.int64)

        return ops.convert_to_tensor(mask)

    def __getitem__(self, index):
        image_path = self.images[index]
        img, tfm, raster_crs = self._load_image(image_path)
        h, w = img.shape[0], img.shape[1]
        sample = {"image": img}

        if self.split == "train":
            mask_path = self.masks[index]
            mask = self._load_mask(mask_path, tfm, raster_crs, (h, w))
            sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _image_id(self, path):
        """Return the image ID."""
        keys = []
        if match := re.search(self.file_regex, path):
            for key in match.group(1).split("_"):
                try:
                    keys.append(int(key))
                except ValueError:
                    keys.append(key)

        return keys

    def _list_files(self, aoi):
        """List all files in a particular AOI."""
        kwargs = {}
        if "{aoi}" in self.directory_glob:
            kwargs["aoi"] = aoi

        product_glob = os.path.join(
            self.root, self.dataset_id, self.split, self.directory_glob
        )
        image_glob = product_glob.format(product=self.image, **kwargs)
        mask_glob = product_glob.format(product=self.mask, **kwargs)
        images = glob.glob(os.path.join(image_glob, self.image_glob), recursive=True)
        masks = glob.glob(os.path.join(mask_glob, self.mask_glob), recursive=True)

        images.sort(key=self._image_id)
        masks.sort(key=self._image_id)

        # Remove images missing masks (SN3) or duplicate images (SN8)
        if self.split == "train":
            images_iter = iter(images)
            images = []
            for mask in masks:
                mask_id = self._image_id(mask)
                for image in images_iter:
                    image_id = self._image_id(image)
                    if image_id == mask_id:
                        images.append(image)
                        break

        return images, masks

    def _verify(self):
        """Verify the integrity of the dataset."""
        self.images = []
        self.masks = []
        root = os.path.join(self.root, self.dataset_id, self.split)
        os.makedirs(root, exist_ok=True)
        for aoi in self.aois:
            images, masks = self._list_files(aoi)
            if images:
                self.images.extend(images)
                self.masks.extend(masks)
                continue

            for tarball, md5 in zip(
                self.tarballs[self.split][aoi], self.md5s[self.split][aoi]
            ):
                path = os.path.join(root, os.path.basename(tarball))
                if os.path.exists(path):
                    extract_archive(path, root)
                    continue

                if not self.download:
                    raise DatasetNotFoundError(self)

                url = self.url.format(dataset_id=self.dataset_id, tarball=tarball)
                aws = which("aws")
                aws("s3", "cp", "--no-sign-request", url, path)
                check_integrity(path, md5 if self.checksum else None)
                extract_archive(path, root)
                images, masks = self._list_files(aoi)
                self.images.extend(images)
                self.masks.extend(masks)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = quantile_normalization(sample["image"][..., :3])
        image = ops.convert_to_numpy(image)

        ncols = 1
        show_mask = "mask" in sample
        show_predictions = "prediction" in sample

        if show_mask:
            mask = ops.convert_to_numpy(sample["mask"])
            ncols += 1

        if show_predictions:
            prediction = ops.convert_to_numpy(sample["prediction"])
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 8, 8))
        axs[0, 0].imshow(image)
        axs[0, 0].axis("off")
        if show_titles:
            axs[0, 0].set_title("Image")

        if show_mask:
            axs[0, 1].imshow(mask, interpolation="none")
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Label")

        if show_predictions:
            axs[0, 2].imshow(prediction, interpolation="none")
            axs[0, 2].axis("off")
            if show_titles:
                axs[0, 2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
