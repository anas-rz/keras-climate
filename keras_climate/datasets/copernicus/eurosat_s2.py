"""Copernicus-Bench EuroSAT-S2 dataset (ported from torchgeo.datasets.copernicus.eurosat_s2)."""

import os

import numpy as np
from keras import ops

from .base import CopernicusBenchBase


class CopernicusBenchEuroSATS2(CopernicusBenchBase):
    """Copernicus-Bench EuroSAT-S2 dataset.

    EuroSAT-S2 is a multi-class land use/land cover classification dataset,
    and is functionally identical to EuroSAT-MS.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://ieeexplore.ieee.org/document/8736785
    * https://ieeexplore.ieee.org/document/8519248
    """

    url = "https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_eurosat_s1s2/eurosat_s2.zip"
    sha256 = "b78f2a2e4e059c0a5d6565335a447465de7468b7f8472835a0d406e6968a15a0"
    zipfile = "eurosat_s2.zip"
    directory = "eurosat_s2"
    filename = "eurosat-{}.txt"
    all_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B09",
        "B10",
        "B11",
        "B12",
        "B8A",
    )
    rgb_bands = ("B04", "B03", "B02")
    classes = (
        "AnnualCrop",
        "HerbaceousVegetation",
        "Industrial",
        "PermanentCrop",
        "River",
        "Forest",
        "Highway",
        "Pasture",
        "Residential",
        "SeaLake",
    )

    def __getitem__(self, index):
        """Return an index within the dataset."""
        file = str(self.files[index]).replace(".jpg", ".tif")
        classname = file.split("_")[0]
        path = os.path.join(self.root, self.directory, "all_imgs", classname, file)
        sample = self._load_image(path)
        sample["label"] = ops.convert_to_tensor(
            np.array(self.classes.index(classname), dtype="int64")
        )

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
