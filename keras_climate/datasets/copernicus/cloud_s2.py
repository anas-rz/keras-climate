"""Copernicus-Bench Cloud-S2 dataset (ported from torchgeo.datasets.copernicus.cloud_s2)."""

import os

from matplotlib.colors import ListedColormap

from .base import CopernicusBenchBase


class CopernicusBenchCloudS2(CopernicusBenchBase):
    """Copernicus-Bench Cloud-S2 dataset.

    Cloud-S2 is a multi-class cloud segmentation dataset derived from
    `CloudSEN12+ <https://www.sciencedirect.com/science/article/pii/S2352340924008163>`_,
    one of the largest Sentinel-2 cloud and cloud shadow detection datasets with
    expert-labeled pixels. We take 25% samples with high-quality labels, and split them
    into 1699/567/551 train/val/test subsets.

    Classes:

    * 0 - Clear: Pixels without cloud and cloud shadow contamination.
    * 1 - Thick Cloud: Opaque clouds that block all the reflected light from
      the Earth's surface.
    * 2 - Thin Cloud: Semitransparent clouds that alter the surface spectral
      signal but still allow to recognize the background. This is the
      hardest class to identify.
    * 3 - Cloud Shadow: Dark pixels where light is occluded by thick or thin
      clouds.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://doi.org/10.1016/j.dib.2024.110852
    """

    url = "https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l1_cloud_s2/cloud_s2.zip"
    sha256 = "38a371b10ad66a6eb702179b2023b1874caf81a1f8dc72ccbfdfd1701acabaef"
    zipfile = "cloud_s2.zip"
    directory = "cloud_s2"
    filename_regex = r"ROI_\d{5}__(?P<date>\d{8}T\d{6})"
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
        "B10",
        "B11",
        "B12",
    )
    rgb_bands = ("B04", "B03", "B02")
    cmap = ListedColormap(["white", "yellow", "green", "red"])
    classes = ("Clear", "Thick Cloud", "Thin Cloud", "Cloud Shadow")

    def __getitem__(self, index):
        file = str(self.files[index]) + ".tif"
        image_path = os.path.join(self.root, self.directory, "s2_toa", file)
        mask_path = os.path.join(self.root, self.directory, "cloud", file)
        sample = self._load_image(image_path) | self._load_mask(mask_path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
