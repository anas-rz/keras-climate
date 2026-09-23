"""Copernicus-Bench LC100Seg-S3 dataset (ported from torchgeo.datasets.copernicus.lc100seg_s3)."""

import glob
import os

import pandas as pd
from keras import ops
from matplotlib.colors import ListedColormap

from ..utils import stack_samples
from .base import CopernicusBenchBase


class CopernicusBenchLC100SegS3(CopernicusBenchBase):
    """Copernicus-Bench LC100Seg-S3 dataset.

    LC100Seg-S3 is a multilabel land use/land cover segmentation dataset
    based on Sentinel-3 OLCI images and CGLS-LC100 land cover maps.
    CGLS-LC100 is a product in the Copernicus Global Land Service (CGLS)
    portfolio and delivers a global 23-class land cover map at 100 m spatial
    resolution.

    This benchmark supports both static (1 image/location) and time series
    (1-4 images/location) modes, the former is used in the original benchmark.

    Classes: 0 Unknown, 20 Shrubs, 30 Herbaceous vegetation, 40 Cultivated and
    managed vegetation/agriculture, 50 Urban/built up, 60 Bare/sparse
    vegetation, 70 Snow and ice, 80 Permanent water bodies, 90 Herbaceous
    wetland, 100 Moss and lichen, 111-116 Closed forest variants, 121-126 Open
    forest variants, 200 Oceans/seas.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://doi.org/10.5281/zenodo.3939049
    """

    url = "https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_lc100_s3/lc100_s3.zip"
    sha256 = "04114d37732aad190b867553d1d45eb094009528521dd54d4ad0045d06e3a51d"
    zipfile = "lc100_s3.zip"
    filename = "multilabel-{}.csv"
    directory = "lc100_s3"
    filename_regex = r"S3[AB]_(?P<date>\d{8}T\d{6})"
    all_bands = (
        "Oa01_radiance",
        "Oa02_radiance",
        "Oa03_radiance",
        "Oa04_radiance",
        "Oa05_radiance",
        "Oa06_radiance",
        "Oa07_radiance",
        "Oa08_radiance",
        "Oa09_radiance",
        "Oa10_radiance",
        "Oa11_radiance",
        "Oa12_radiance",
        "Oa13_radiance",
        "Oa14_radiance",
        "Oa15_radiance",
        "Oa16_radiance",
        "Oa17_radiance",
        "Oa18_radiance",
        "Oa19_radiance",
        "Oa20_radiance",
        "Oa21_radiance",
    )
    rgb_bands = ("Oa08_radiance", "Oa06_radiance", "Oa04_radiance")
    cmap = ListedColormap(
        [
            "#282828",
            "#ffbb22",
            "#ffff4c",
            "#f096ff",
            "#fa0000",
            "#b4b4b4",
            "#f0f0f0",
            "#0032c8",
            "#0096a0",
            "#fae6a0",
            "#58481f",
            "#009900",
            "#70663e",
            "#00cc00",
            "#4e751f",
            "#007800",
            "#666000",
            "#8db400",
            "#8d7400",
            "#a0dc00",
            "#929900",
            "#648c00",
            "#000080",
        ]
    )
    classes = (
        "Unknown",
        "Shrubs",
        "Herbaceous vegetation",
        "Cultivated and managed vegetation / agriculture",
        "Urban / built up",
        "Bare / sparse vegetation",
        "Snow and ice",
        "Permanent water bodies",
        "Herbaceous wetland",
        "Moss and lichen",
        "Closed forest, evergreen needle leaf",
        "Closed forest, evergreen broad leaf",
        "Closed forest, deciduous needle leaf",
        "Closed forest, deciduous broad leaf",
        "Closed forest, mixed",
        "Closed forest, not matching any of the other definitions",
        "Open forest, evergreen needle leaf",
        "Open forest, evergreen broad leaf",
        "Open forest, deciduous needle leaf",
        "Open forest, deciduous broad leaf",
        "Open forest, mixed",
        "Open forest, not matching any of the other definitions",
        "Oceans, seas",
    )
    class_map = {
        0: 0,
        20: 1,
        30: 2,
        40: 3,
        50: 4,
        60: 5,
        70: 6,
        80: 7,
        90: 8,
        100: 9,
        111: 10,
        112: 11,
        113: 12,
        114: 13,
        115: 14,
        116: 15,
        121: 16,
        122: 17,
        123: 18,
        124: 19,
        125: 20,
        126: 21,
        200: 22,
    }

    def __init__(
        self,
        root="data",
        split="train",
        mode="static",
        bands=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new CopernicusBenchLC100SegS3 instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: One of 'static' or 'time-series'.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its
                target as entry and returns a transformed version.
            download: If True, download dataset and store it in the root
                directory.
            checksum: If True, verify the checksum of the downloaded files
                (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.mode = mode
        super().__init__(root, split, bands, transforms, download, checksum)
        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath)
        if mode == "static":
            filepath = os.path.join(root, self.directory, f"static_fnames-{split}.csv")
            self.static_files = pd.read_csv(filepath, header=None)

    def __getitem__(self, index):
        """Return an index within the dataset."""
        row = self.files.iloc[index].values
        if self.mode == "static":
            pid, file = self.static_files.iloc[index]
            path = os.path.join(self.root, self.directory, "s3_olci", pid, file)
            sample = self._load_image(path)
        elif self.mode == "time-series":
            pid = row[0]
            paths = os.path.join(self.root, self.directory, "s3_olci", pid, "*.tif")
            samples = [self._load_image(path) for path in sorted(glob.glob(paths))]
            sample = stack_samples(samples)
        else:
            raise ValueError(f"Invalid mode: {self.mode!r}")

        path = os.path.join(self.root, self.directory, "lc100", f"{pid}.tif")
        sample = sample | self._load_mask(path)
        mask = ops.convert_to_numpy(sample["mask"])
        remapped = mask.copy()
        for before, after in self.class_map.items():
            remapped[mask == before] = after
        sample["mask"] = ops.cast(ops.convert_to_tensor(remapped), self.dtype)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
