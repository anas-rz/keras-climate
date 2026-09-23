"""DL4GAMAlps Dataset (ported from torchgeo.datasets.dl4gam)."""

import pathlib

import numpy as np
import pandas as pd
from keras import ops

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive, download_url, extract_archive, lazy_import


class DL4GAMAlps(NonGeoDataset):
    r"""A Multi-modal Dataset for Glacier Mapping (Segmentation) in the European Alps.

    The dataset consists of Sentinel-2 images from 2015 (mainly), 2016 and
    2017, and binary segmentation masks for glaciers, based on an inventory
    built by glaciology experts (`Paul et al. 2020
    <https://doi.org/10.1594/PANGAEA.909133>`_).

    Dataset features:

    * Sentinel-2 images (all bands, including cloud and shadow masks which
      can be used for loss masking)
    * glacier mask (0: no glacier, 1: glacier)
    * debris mask (0: no debris, 1: debris)
    * DEM (Copernicus GLO-30) + five derived features: slope, aspect,
      terrain ruggedness index, planform and profile curvatures
    * dh/dt (surface elevation change) map over 2010-2015
    * v (surface velocity) map over 2015

    For more details check also: https://huggingface.co/datasets/dcodrut/dl4gam_alps

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.22541/essoar.173557607.70204641/v1

    .. note::

        This dataset requires the following additional libraries to be
        installed:

        * `xarray <https://pypi.org/project/xarray/>`_
        * `netcdf4 <https://pypi.org/project/netCDF4/>`_
          or `h5netcdf <https://pypi.org/project/h5netcdf/>`_
    """

    url = "https://huggingface.co/datasets/dcodrut/dl4gam_alps/resolve/7d20ca8a2b30c5518e086ffaa5ce37e6a66c42c1/data"
    download_metadata = {
        "dataset_small": {
            "url": f"{url}/patches/inv_r_128_s_128.tar.gz",
            "checksum": "3e69c47c6ff5106cd4ffaa6bb2caaaef",
        },
        "dataset_large": {
            "url": f"{url}/patches/inv_r_128_s_64.tar.gz",
            "checksum": "06e85a6a9e3dc6b3cdb07f928e832bc8",
        },
        "splits_csv": {
            "url": f"{url}/map_all_splits_all_folds.csv",
            "checksum": "862355c5c3482271dd171d31c70551b3",
        },
    }

    rgb_bands = ("B4", "B3", "B2")
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
    rgb_nir_swir_bands = ("B4", "B3", "B2", "B8", "B11")  # the subset used in the paper

    valid_extra_features = (
        "dem",
        "slope",
        "aspect",
        "planform_curvature",
        "profile_curvature",
        "terrain_ruggedness_index",
        "dhdt",
        "v",
    )
    valid_splits = ("train", "val", "test")
    valid_versions = ("small", "large")
    valid_cv_iters = (1, 2, 3, 4, 5)

    def __init__(
        self,
        root="data",
        split="train",
        cv_iter=1,
        version="small",
        bands=rgb_nir_swir_bands,
        extra_features=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize the dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            cv_iter: one of 1, 2, 3, 4, 5 (for the five-fold geographical
                cross-validation scheme)
            version: one of "small" or "large" (controls the sampling
                overlap)
            bands: the Sentinel-2 bands to use as input (default: RGB + NIR
                + SWIR)
            extra_features: additional features to include (default: None;
                see the class attribute for the available)
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: if any parameters are invalid.
            DatasetNotFoundError: if dataset is not found and *download* is
                False.
            DependencyNotFoundError: if xarray is not installed.
        """
        lazy_import("xarray")

        self.root = pathlib.Path(root)
        self.split = split
        self.cv_iter = cv_iter
        self.version = version
        self.bands = bands
        self.extra_features = extra_features
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        # sanity checks
        assert split in self.valid_splits, f"Split {split} not in: {self.valid_splits}"
        assert cv_iter in self.valid_cv_iters, f"Cross-validation iteration {cv_iter} not in: {self.valid_cv_iters}"
        assert version in self.valid_versions, f"Version {version} not in: {self.valid_versions}"
        for band in bands:
            assert band in self.all_bands, f"Band {band} not in: {self.all_bands}"
        if extra_features:
            for feature in extra_features:
                assert feature in self.valid_extra_features, f"Feature {feature} not in: {self.valid_extra_features}"

        # set the local file paths
        label = f"dataset_{version}"
        self.fp_archive = self.root / f"{label}.tar.gz"
        self.dir_patches = self.root / label
        self.fp_splits_csv = self.root / "splits.csv"

        # get the corresponding urls and checksums
        self.url_dataset = self.download_metadata[label]["url"]
        self.md5_dataset = self.download_metadata[label]["checksum"]
        self.url_csv_splits = self.download_metadata["splits_csv"]["url"]
        self.md5_csv_splits = self.download_metadata["splits_csv"]["checksum"]

        self._verify()
        self._prepare_files()

    def __len__(self):
        return len(self.fp_patches)

    def __getitem__(self, index):
        xr = lazy_import("xarray")

        with xr.open_dataset(self.fp_patches[index], decode_coords="all", mask_and_scale=True) as nc:
            # extract the S2 image and masks from the netcdf file
            all_band_names = nc.band_data.long_name
            idx_img = [all_band_names.index(b) for b in self.bands]
            # image is (band, y, x) -> transpose to (y, x, band) (channels-last)
            image = nc.band_data.isel(band=idx_img).values.astype(np.float32)
            image = np.transpose(image, (1, 2, 0))
            id_cloud_mask = all_band_names.index("CLOUDLESS_MASK")
            mask_clouds_and_shadows = ~(nc.band_data.isel(band=id_cloud_mask).values == 1)
            sample = {
                "image": ops.convert_to_tensor(image),
                "mask_glacier": ops.convert_to_tensor(~np.isnan(nc.mask_all_g_id.values)),
                "mask_debris": ops.convert_to_tensor(nc.mask_debris.values == 1),
                "mask_clouds_and_shadows": ops.convert_to_tensor(mask_clouds_and_shadows),
            }

            # extract the additional features if needed
            if self.extra_features:
                for feature in self.extra_features:
                    assert feature in nc, f"Feature {feature} not found in the netcdf file"
                    vals = nc[feature].values.astype(np.float32)

                    # impute the missing values with the mean
                    # or zero (for dh/dt and surface velocity)
                    v_fill = 0.0 if feature in ("dhdt", "v") else np.nanmean(vals)
                    vals[np.isnan(vals)] = v_fill

                    sample[feature] = ops.convert_to_tensor(vals)

            if self.transforms is not None:
                sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        if self.dir_patches.exists() and self.fp_splits_csv.exists():
            return

        # check if the archive exists
        if self.fp_archive.exists():
            extract_archive(self.fp_archive, self.dir_patches)
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self):
        """Download the patches and the csv with the splits."""
        # download and extract the archive
        download_and_extract_archive(
            url=self.url_dataset,
            download_root=self.root,
            extract_root=self.dir_patches,
            filename=self.fp_archive.name,
            md5=self.md5_dataset if self.checksum else None,
        )

        # download the splits csv
        download_url(
            url=self.url_csv_splits, root=self.root, filename="splits.csv", md5=self.md5_csv_splits if self.checksum else None
        )

    def _prepare_files(self):
        """Prepare the files for the dataset."""
        # prepare the paths to the patches
        self.fp_patches = sorted(self.dir_patches.rglob("*.nc"))

        # get the glacier IDs of the current split of the cross-validation
        self.df_splits = pd.read_csv(self.fp_splits_csv)
        fold_name = f"fold_{self.split if self.split != 'val' else 'valid'}"
        idx = self.df_splits[f"split_{self.cv_iter}"] == fold_name
        glacier_ids = list(self.df_splits.loc[idx, "entry_id"])

        # filter the patches to keep only the ones corresponding to the current split
        self.fp_patches = [fp for fp in self.fp_patches if fp.parent.name in glacier_ids]

    def plot(self, sample, show_titles=True, suptitle=None, clip_extrema=True):
        """Plot a sample from the dataset.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        import matplotlib.pyplot as plt

        # we expect the RGB bands to be present
        if not {"B4", "B3", "B2"}.issubset(set(self.bands)):
            raise RGBBandsMissingError()
        nir_and_swir_present = {"B8", "B11"}.issubset(set(self.bands))

        # prepare the RGB image and the masks
        idx_rgb = [self.bands.index(b) for b in ["B4", "B3", "B2"]]
        rgb_img = ops.take(sample["image"], np.array(idx_rgb), axis=-1)
        images = {
            "RGB Image": rgb_img,
            "Glacier Mask": sample["mask_glacier"],
            "Debris Mask": sample["mask_debris"],
            "Clouds and Shadows Mask": sample["mask_clouds_and_shadows"],
        }

        # add the SWIR-NIR-R image if the bands are present
        if nir_and_swir_present:
            idx_swir_nir_r = [self.bands.index(b) for b in ["B11", "B8", "B4"]]
            swir_nir_r_img = ops.take(sample["image"], np.array(idx_swir_nir_r), axis=-1)
            images["SWIR-NIR-R Image"] = swir_nir_r_img

        # add the extra features if present
        for extra_v, title in (
            ("prediction", "Prediction"),
            ("dem", "DEM"),
            ("slope", "Slope"),
            ("aspect", "Aspect"),
            ("planform_curvature", "Planform Curvature"),
            ("profile_curvature", "Profile Curvature"),
            ("terrain_ruggedness_index", "Terrain Ruggedness Index"),
            ("dhdt", "dh/dt"),
            ("v", "Surface Velocity"),
        ):
            if extra_v in sample:
                images[title] = sample[extra_v]

        cmaps = {
            "RGB Image": None,
            "SWIR-NIR-R Image": None,
            "Glacier Mask": "gray",
            "Prediction": "gray",
            "Debris Mask": "gray",
            "Clouds and Shadows Mask": "gray",
            "DEM": "terrain",
            "Slope": "magma",
            "Aspect": "jet",
            "Planform Curvature": "magma",
            "Profile Curvature": "magma",
            "Terrain Ruggedness Index": "magma",
            "dh/dt": "seismic_r",
            "Surface Velocity": "magma",
        }

        # build the figure
        n_imgs = len(images)
        ncols = 4 if n_imgs <= 8 else 5
        nrows = int(np.ceil(n_imgs / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

        for ax, k in zip(np.array(axs).flat, images):
            img = ops.convert_to_numpy(images[k])
            cmap = cmaps[k]

            # clip the extrema 5% of the values if needed
            if clip_extrema and k not in ["Glacier Mask", "Prediction", "Debris Mask", "Clouds and Shadows Mask"]:
                q_lim_clip = 0.025
                img = np.clip(img, np.quantile(img, q_lim_clip), np.quantile(img, 1 - q_lim_clip))

            vmin, vmax = np.min(img), np.max(img)
            # scale the images to [0,1]
            if k in ["RGB Image", "SWIR-NIR-R Image"]:
                img = (img - vmin) / (vmax - vmin)

            if k == "dh/dt":  # diverging colormap for the dh/dt, make it symmetric
                max_abs = max(abs(vmin), abs(vmax))
                vmin, vmax = -max_abs, max_abs

            ax.imshow(img, cmap=cmap, interpolation="none", vmin=vmin, vmax=vmax)
            if show_titles:
                ax.set_title(k)

        # disable the axes for all plots, including the empty plots
        for ax in np.array(axs).flat:
            ax.axis("off")

        if suptitle:
            fig.suptitle(suptitle)

        return fig
