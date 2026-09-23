"""Canadian Building Footprints dataset (ported from torchgeo.datasets.cbf)."""

import os

from .errors import DatasetNotFoundError
from .geo import VectorDataset
from .utils import check_integrity, download_and_extract_archive


class CanadianBuildingFootprints(VectorDataset):
    """Canadian Building Footprints dataset.

    The `Canadian Building Footprints
    <https://github.com/Microsoft/CanadianBuildingFootprints>`__ dataset contains
    11,842,186 computer generated building footprints in all Canadian provinces and
    territories in GeoJSON format. This data is freely available for download and use.
    """

    url = "https://usbuildingdata.blob.core.windows.net/canadian-buildings-v2/"
    provinces_territories = (
        "Alberta",
        "BritishColumbia",
        "Manitoba",
        "NewBrunswick",
        "NewfoundlandAndLabrador",
        "NorthwestTerritories",
        "NovaScotia",
        "Nunavut",
        "Ontario",
        "PrinceEdwardIsland",
        "Quebec",
        "Saskatchewan",
        "YukonTerritory",
    )
    filename_glob = "*.geojson"

    md5s = (
        "8b4190424e57bb0902bd8ecb95a9235b",
        "fea05d6eb0006710729c675de63db839",
        "adf11187362624d68f9c69aaa693c46f",
        "44269d4ec89521735389ef9752ee8642",
        "65dd92b1f3f5f7222ae5edfad616d266",
        "346d70a682b95b451b81b47f660fd0e2",
        "bd57cb1a7822d72610215fca20a12602",
        "c1f29b73cdff9a6a9dd7d086b31ef2cf",
        "76ba4b7059c5717989ce34977cad42b2",
        "2e4a3fa47b3558503e61572c59ac5963",
        "9ff4417ae00354d39a0cf193c8df592c",
        "a51078d8e60082c7d3a3818240da6dd5",
        "c11f3bd914ecabd7cac2cb2871ec0261",
    )

    def __init__(
        self,
        paths="data",
        crs=None,
        res=(0.00001, 0.00001),
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres)
                format. If a single float is provided, it is used for both
                the x and y resolution.
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.paths = paths
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        super().__init__(paths, crs, res, transforms)

    def _check_integrity(self):
        """Check integrity of dataset."""
        paths = self.paths
        for prov_terr, md5 in zip(self.provinces_territories, self.md5s):
            filepath = os.path.join(paths, prov_terr + ".zip")
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

    def _download(self):
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        paths = self.paths
        for prov_terr, md5 in zip(self.provinces_territories, self.md5s):
            download_and_extract_archive(
                self.url + prov_terr + ".zip", paths, md5=md5 if self.checksum else None
            )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt
        from keras import ops

        image = ops.convert_to_numpy(sample["mask"])
        ncols = 1

        showing_prediction = "prediction" in sample
        if showing_prediction:
            pred = ops.convert_to_numpy(sample["prediction"])
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4, 4))

        if showing_prediction:
            axs[0].imshow(image)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(image)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
