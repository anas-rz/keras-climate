"""MMFlood dataset (ported from torchgeo.datasets.mmflood)."""

import os
from glob import glob

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError
from .geo import IntersectionDataset, RasterDataset
from .utils import download_url, extract_archive


def _concat_channels(samples):
    """Collate function that concatenates ``image``/``mask`` on the channel axis.

    Unlike :func:`keras_climate.datasets.utils.concat_samples` (which
    concatenates every shared key along axis 0), this keeps the
    channels-last convention used throughout keras_climate: ``image`` and
    ``mask`` tensors from each dataset are concatenated along the last
    (channel) axis, while other shared keys (``bounds``, ``transform``) are
    taken from the first dataset since they describe the same spatiotemporal
    query.
    """
    keys = []
    for sample in samples:
        for key in sample:
            if key not in keys:
                keys.append(key)

    collated = {}
    for key in keys:
        values = [sample[key] for sample in samples if key in sample]
        if key in ("image", "mask") and len(values) > 1:
            collated[key] = ops.concatenate(values, axis=-1)
        else:
            collated[key] = values[0]
    return collated


class MMFloodComponent(RasterDataset):
    """Base component for MMFlood dataset."""

    def __init__(
        self,
        subfolders,
        content,
        root="data",
        crs=None,
        res=None,
        transforms=None,
        cache=False,
        time_series=False,
    ):
        """Initialize MMFloodComponent dataset instance.

        Args:
            subfolders: list of directories to be loaded
            content: specifies which component to load, one of "s1_raw",
                "DEM", "hydro", "mask"
            root: root directory where dataset can be found
            crs: CRS to warp to (defaults to the CRS of the first file
                found)
            res: resolution of the dataset in units of CRS in (xres, yres)
                format
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            time_series: if True, stack data along the time series
                dimension
        """
        self.content = content
        self.is_image = content != "mask"
        paths = []
        for s in subfolders:
            paths += glob(os.path.join(root, "**", f"{s}*-*", self.content, "*.tif"))
        paths = sorted(paths)
        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )


class MMFlood(IntersectionDataset):
    """MMFlood dataset.

    `MMFlood <https://huggingface.co/datasets/links-ads/mmflood>`__ dataset
    is a multimodal flood delineation dataset. Sentinel-1 data is matched
    with masks and DEM data for all available tiles. If hydrography maps are
    loaded, only a subset of the dataset is loaded, since only 1,012
    Sentinel-1 tiles have a corresponding hydrography map. Some Sentinel-1
    tiles have missing data, which are automatically set to 0. Corresponding
    pixels in masks are set to 255 and should be ignored in performance
    computation.

    Dataset features:

    * 1,748 Sentinel-1 tiles of varying pixel dimensions
    * multimodal dataset
    * 95 flood events from 42 different countries
    * includes DEMs
    * includes hydrography maps (available for 1,012 tiles out of 1,748)
    * flood delineation maps (ground truth) is obtained from Copernicus EMS

    Dataset classes:

    0. no flood
    1. flood

    If you use this dataset in your research, please cite the following
    paper:

    * https://doi.org/10.1109/ACCESS.2022.3205419
    """

    url = "https://huggingface.co/datasets/links-ads/mmflood/resolve/24ca097306c9e50ad0711903c11e1ba13ea1bedc/"
    _ignore_index = 255
    _nparts = 11

    metadata = {
        "part_file": "activations.tar.{part}.gz.part",
        "filename": "activations.tar.gz",
        "directory": "activations",
        "metadata_file": "activations.json",
    }
    _splits = {"train", "val", "test"}
    _md5 = {
        "activations.json": "de33a3ac7e55a0051ada21cbdfbb4745",
        "activations.tar.gz": "3cd4c4fe7506aa40263f74639d85ccce",
        "activations.tar.000.gz.part": "a8424653edca6e79999831bdda53d4dc",
        "activations.tar.001.gz.part": "517def8760d3ce86885c7600c77a1d6c",
        "activations.tar.002.gz.part": "6797b97121f5b98ff58fde7491f584b2",
        "activations.tar.003.gz.part": "e69d2a6b1746ef869d1da4d22018a71a",
        "activations.tar.004.gz.part": "0ccf7ea69ea6c0e88db1b1015ec3361e",
        "activations.tar.005.gz.part": "8ef6765afe20f254b1e752d7a2742fda",
        "activations.tar.006.gz.part": "3f330a44b66511b7a95f4a555f8b793a",
        "activations.tar.007.gz.part": "1d2046b5f3c473c3681a05dc94b29b86",
        "activations.tar.008.gz.part": "f386b5acf78f8ae34592404c6c7ec43c",
        "activations.tar.009.gz.part": "dd5317a3c0d33de815beadb9850baa38",
        "activations.tar.010.gz.part": "5a14a7e3f916c5dcf288c2ca88daf4d0",
    }

    def __init__(
        self,
        root="data",
        crs=None,
        res=None,
        split="train",
        include_dem=False,
        include_hydro=False,
        transforms=None,
        download=False,
        checksum=True,
        cache=False,
        time_series=False,
    ):
        """Initialize a new MMFlood dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: CRS to warp to (defaults to the CRS of the first file
                found)
            res: resolution of the dataset in units of CRS in (xres, yres)
                format
            split: train/val/test split to load
            include_dem: If True, DEM data is concatenated after Sentinel-1
                bands.
            include_hydro: If True, hydrography data is concatenated as last
                channel. Only a smaller subset of the original dataset is
                loaded in this case.
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)
            cache: if True, cache file handle to speed up repeated sampling
            time_series: if True, stack data along the time series dimension

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
            AssertionError: If *split* is invalid.
        """
        assert split in self._splits

        self.root = root
        self.split = split
        self.include_dem = include_dem
        self.include_hydro = include_hydro
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        # Verify integrity of the dataset
        self._verify()
        self.metadata_df = pd.read_json(
            os.path.join(self.root, self.metadata["metadata_file"])
        ).transpose()

        split_subfolders = self.metadata_df[
            self.metadata_df["subset"] == self.split
        ].index.tolist()
        self.image = MMFloodComponent(
            split_subfolders,
            "s1_raw",
            root,
            crs,
            res,
            cache=cache,
            time_series=time_series,
        )
        if include_dem:
            dem = MMFloodComponent(
                split_subfolders,
                "DEM",
                root,
                crs,
                res,
                cache=cache,
                time_series=time_series,
            )
            self.image = IntersectionDataset(
                self.image, dem, collate_fn=_concat_channels
            )
            self.image.index = dem.index
        if include_hydro:
            hydro = MMFloodComponent(
                split_subfolders,
                "hydro",
                root,
                crs,
                res,
                cache=cache,
                time_series=time_series,
            )
            self.image = IntersectionDataset(
                self.image, hydro, collate_fn=_concat_channels
            )
            self.image.index = hydro.index
        self.mask = MMFloodComponent(
            split_subfolders,
            "mask",
            root,
            crs,
            res,
            cache=cache,
            time_series=time_series,
        )

        super().__init__(
            self.image, self.mask, collate_fn=_concat_channels, transforms=transforms
        )
        self.index = self.image.index

    def _merge_tar_files(self):
        """Merge part tar gz files."""
        dst_filename = self.metadata["filename"]
        dst_path = os.path.join(self.root, dst_filename)

        print("Merging separate part files...")
        with open(dst_path, "wb") as dst_fp:
            for idx in range(self._nparts):
                part_filename = f"activations.tar.{idx:03}.gz.part"
                part_path = os.path.join(self.root, part_filename)
                print(f"Processing file {part_path!s}")

                with open(part_path, "rb") as part_fp:
                    dst_fp.write(part_fp.read())

    def __getitem__(self, index):
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice."""
        data = super().__getitem__(index)

        image_np = ops.convert_to_numpy(data["image"])
        missing_data = np.isnan(image_np).any(axis=-1)
        image_np = np.where(missing_data[..., None], 0, image_np)

        mask_np = ops.convert_to_numpy(data["mask"]).astype("int64")
        mask_np = np.where(missing_data, self._ignore_index, mask_np)

        data["image"] = ops.convert_to_tensor(image_np.astype("float32"))
        data["mask"] = ops.convert_to_tensor(mask_np)
        return data

    def _download(self):
        """Download the dataset."""

        def _check_and_download(filename, url):
            path = os.path.join(self.root, filename)
            if not os.path.exists(path):
                md5 = self._md5[filename] if self.checksum else None
                download_url(url, self.root, filename, md5)

        filename = self.metadata["filename"]
        filepath = os.path.join(self.root, filename)
        if not os.path.exists(filepath):
            for idx in range(self._nparts):
                part_file = f"activations.tar.{idx:03}.gz.part"
                url = self.url + part_file

                _check_and_download(part_file, url)

        _check_and_download(
            self.metadata["metadata_file"], self.url + self.metadata["metadata_file"]
        )

    def _extract(self):
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.metadata["filename"])
        if str(filepath).endswith(".tar.gz"):
            extract_archive(filepath)

    def _verify(self):
        """Verify the integrity of the dataset."""
        dirpath = os.path.join(self.root, self.metadata["directory"])
        metadata_filepath = os.path.join(self.root, self.metadata["metadata_file"])
        # Check if both metadata file and directory exist
        if os.path.isdir(dirpath) and os.path.isfile(metadata_filepath):
            return
        if not self.download:
            raise DatasetNotFoundError(self)
        self._download()
        self._merge_tar_files()
        self._extract()

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        show_mask = "mask" in sample
        image = ops.convert_to_numpy(sample["image"])[..., [0, 1]]
        ncols = 1
        show_predictions = "prediction" in sample
        axs_idx = 1
        if self.include_dem:
            dem_idx = -2 if self.include_hydro else -1
            dem = ops.convert_to_numpy(sample["image"])[..., dem_idx]
            ncols += 1
        if self.include_hydro:
            hydro = ops.convert_to_numpy(sample["image"])[..., -1]
            ncols += 1
        if show_mask:
            mask = ops.convert_to_numpy(sample["mask"]).copy()
            # Set ignore_index values to 0
            mask[mask == self._ignore_index] = 0
            ncols += 1
        if show_predictions:
            pred = ops.convert_to_numpy(sample["prediction"])
            ncols += 1

        # Compute False Color image, from Sentinel1 plot function
        co_polarization = image[..., 0]  # transmit == receive
        cross_polarization = image[..., 1]  # transmit != receive
        ratio = co_polarization / cross_polarization

        # https://gis.stackexchange.com/a/400780/123758
        co_polarization = np.clip(co_polarization / 0.3, a_min=0, a_max=1)
        cross_polarization = np.clip(cross_polarization / 0.05, a_min=0, a_max=1)
        ratio = np.clip(ratio / 25, a_min=0, a_max=1)

        image = np.stack((co_polarization, cross_polarization, ratio), axis=-1)

        # Generate the figure
        fig, axs = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        axs_list = axs if ncols > 1 else [axs]
        axs_list[0].imshow(image)
        axs_list[0].axis("off")
        axs_idx = 1
        if self.include_dem:
            axs_list[axs_idx].imshow(dem, cmap="gray")
            axs_list[axs_idx].axis("off")
            axs_idx += 1
        if self.include_hydro:
            axs_list[axs_idx].imshow(hydro, cmap="gray")
            axs_list[axs_idx].axis("off")
            axs_idx += 1
        if show_mask:
            axs_list[axs_idx].imshow(mask, cmap="gray")
            axs_list[axs_idx].axis("off")
            axs_idx += 1
        if show_predictions:
            axs_list[axs_idx].imshow(pred, cmap="gray")
            axs_list[axs_idx].axis("off")

        if show_titles:
            axs_list[0].set_title("Image")
            axs_idx = 1
            if self.include_dem:
                axs_list[axs_idx].set_title("DEM")
                axs_idx += 1
            if self.include_hydro:
                axs_list[axs_idx].set_title("Hydrography Map")
                axs_idx += 1
            if show_mask:
                axs_list[axs_idx].set_title("Mask")
                axs_idx += 1
            if show_predictions:
                axs_list[axs_idx].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
