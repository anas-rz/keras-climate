"""Copernicus-Bench Flood-S1 dataset (ported from torchgeo.datasets.copernicus.flood_s1)."""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
import rasterio as rio
from keras import ops
from matplotlib.colors import ListedColormap
from pyproj import Transformer

from ..utils import disambiguate_timestamp
from .base import CopernicusBenchBase


class CopernicusBenchFloodS1(CopernicusBenchBase):
    """Copernicus-Bench Flood-S1 dataset.

    Flood-S1 is a flood segmentation dataset extracted from a large flood
    mapping dataset Kuro Siwo.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://arxiv.org/abs/2311.12056
    """

    url = "https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l3_flood_s1/flood_s1.zip"
    sha256 = "aecaa4437dcc6575900531d7ade8752c1883d721782020d188b9d1df626762fd"
    zipfile = "flood_s1.zip"
    directory = "flood_s1"
    filename = "grid_dict_{}.json"
    filename_regex = r".{18}_(?P<date>\d{8})"
    date_format = "%Y%m%d"
    all_bands = ("VV", "VH")
    rgb_bands = ("VV", "VH")
    cmap = ListedColormap(["black", "cyan", "magenta"])
    classes = ("No Water", "Permanent Waters", "Floods")

    def __init__(
        self,
        root="data",
        split="train",
        mode=1,
        bands=None,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new CopernicusBenchFloodS1 instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: Number of pre-flood images, 1 or 2.
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
        self.root = root
        self.split = split
        self.mode = mode
        self.bands = bands or self.all_bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        filepath = os.path.join(root, self.directory, self.filename.format(split))
        with open(filepath) as f:
            self.metadata = json.load(f)
        self.files = pd.Series(sorted(self.metadata.keys()))

    def __getitem__(self, index):
        """Return an index within the dataset."""
        key = self.files[index]
        path = self.metadata[key]["path"]
        directory = os.path.join(self.root, self.directory, "data", path)
        mask_path = glob.glob(os.path.join(directory, "MK0_MLU*.tif"))[0]
        sample = self._load_image(directory) | self._load_mask(mask_path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path):
        """Load an image and metadata.

        Returns:
            An image sample, with the pseudo time series of pre/post-flood
            images stacked channels-last (``T x H x W x C``).
        """
        images = []
        times = []
        ptypes = ["SL1", "MS1"]
        if self.mode == 2:
            ptypes.insert(0, "SL2")

        filepath = None
        for ptype in ptypes:
            image = []
            for band in self.bands:
                # Band (every band)
                filepath = glob.glob(os.path.join(path, f"{ptype}_I{band}_*.tif"))[0]
                with rio.open(filepath) as f:
                    image.append(f.read(1).astype(np.float32))

            # (C, H, W) -> (H, W, C)
            image = np.stack(image, axis=-1)
            # Image (every ptype)
            images.append(image)

            # Time (every ptype)
            if (
                match := re.match(self.filename_regex, os.path.basename(filepath))
            ) and "date" in match.groupdict():
                date_str = match.group("date")
                mint, maxt = disambiguate_timestamp(date_str, self.date_format)
                time = (mint.timestamp() + maxt.timestamp()) / 2
                times.append(time)

        # Location (only once)
        with rio.open(filepath) as f:
            x = (f.bounds.left + f.bounds.right) / 2
            y = (f.bounds.bottom + f.bounds.top) / 2
            transformer = Transformer.from_crs(f.crs, "epsg:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)

        return {
            "image": ops.convert_to_tensor(np.stack(images)),
            "lat": ops.convert_to_tensor(lat),
            "lon": ops.convert_to_tensor(lon),
            "time": ops.convert_to_tensor(np.array(times, dtype="float64")),
        }
