"""Copernicus-Bench EuroSAT-S1 dataset (ported from torchgeo.datasets.copernicus.eurosat_s1)."""

import os

import numpy as np
from keras import ops

from .base import CopernicusBenchBase


class CopernicusBenchEuroSATS1(CopernicusBenchBase):
    """Copernicus-Bench EuroSAT-S1 dataset.

    EuroSAT-S1 is a multi-class land use/land cover classification dataset,
    and is functionally identical to EuroSAT-SAR.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://doi.org/10.1109/JSTARS.2024.3493237
    """

    url = "https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_eurosat_s1s2/eurosat_s1.zip"
    sha256 = "aa52080b365c45b21097b0d3d190fc5953ad6639d211023c9689be57aed636cf"
    zipfile = "eurosat_s1.zip"
    directory = "eurosat_s1"
    filename = "eurosat-{}.txt"
    all_bands = ("VV", "VH")
    rgb_bands = ("VV", "VH")
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
