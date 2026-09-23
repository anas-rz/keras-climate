"""CMS Global Mangrove Canopy dataset (ported from torchgeo.datasets.cms_mangrove_canopy)."""

import os

from keras import ops

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import check_integrity, extract_archive


class CMSGlobalMangroveCanopy(RasterDataset):
    """CMS Global Mangrove Canopy dataset.

    The `CMS Global Mangrove Canopy dataset
    <https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-cms-global-map-mangrove-canopy-1665-1.3>`_
    consists of a single band map at 30m resolution of either aboveground biomass (agb),
    basal area weighted height (hba95), or maximum canopy height (hmax95).

    The dataset needs to be manually downloaded from the above link, where you can make
    an account and subsequently download the dataset.
    """

    is_image = False

    filename_regex = r"""^
        (?P<mangrove>[A-Za-z]{8})
        _(?P<variable>[a-z0-9]*)
        _(?P<country>[A-Za-z][^.]*)
    """

    zipfile = "CMS_Global_Map_Mangrove_Canopy_1665.zip"
    md5 = "3e7f9f23bf971c25e828b36e6c5496e3"

    all_countries = (
        "AndamanAndNicobar",
        "Angola",
        "Anguilla",
        "AntiguaAndBarbuda",
        "Aruba",
        "Australia",
        "Bahamas",
        "Bahrain",
        "Bangladesh",
        "Barbados",
        "Belize",
        "Benin",
        "Brazil",
        "BritishVirginIslands",
        "Brunei",
        "Cambodia",
        "Cameroon",
        "CarribeanCaymanIslands",
        "China",
        "Colombia",
        "Comoros",
        "CostaRica",
        "Cote",
        "CoteDivoire",
        "CotedIvoire",
        "Cuba",
        "DemocraticRepublicOfCongo",
        "Djibouti",
        "DominicanRepublic",
        "EcuadorWithGalapagos",
        "Egypt",
        "ElSalvador",
        "EquatorialGuinea",
        "Eritrea",
        "EuropaIsland",
        "Fiji",
        "Fiji2",
        "FrenchGuiana",
        "FrenchGuyana",
        "FrenchPolynesia",
        "Gabon",
        "Gambia",
        "Ghana",
        "Grenada",
        "Guadeloupe",
        "Guam",
        "Guatemala",
        "Guinea",
        "GuineaBissau",
        "Guyana",
        "Haiti",
        "Hawaii",
        "Honduras",
        "HongKong",
        "India",
        "Indonesia",
        "Iran",
        "Jamaica",
        "Japan",
        "Kenya",
        "Liberia",
        "Macau",
        "Madagascar",
        "Malaysia",
        "Martinique",
        "Mauritania",
        "Mayotte",
        "Mexico",
        "Micronesia",
        "Mozambique",
        "Myanmar",
        "NewCaledonia",
        "NewZealand",
        "Newzealand",
        "Nicaragua",
        "Nigeria",
        "NorthernMarianaIslands",
        "Oman",
        "Pakistan",
        "Palau",
        "Panama",
        "PapuaNewGuinea",
        "Peru",
        "Philipines",
        "PuertoRico",
        "Qatar",
        "ReunionAndMauritius",
        "SaintKittsAndNevis",
        "SaintLucia",
        "SaintVincentAndTheGrenadines",
        "Samoa",
        "SaudiArabia",
        "Senegal",
        "Seychelles",
        "SierraLeone",
        "Singapore",
        "SolomonIslands",
        "Somalia",
        "Somalia2",
        "Soudan",
        "SouthAfrica",
        "SriLanka",
        "Sudan",
        "Suriname",
        "Taiwan",
        "Tanzania",
        "Thailand",
        "TimorLeste",
        "Togo",
        "Tonga",
        "TrinidadAndTobago",
        "TurksAndCaicosIslands",
        "Tuvalu",
        "UnitedArabEmirates",
        "UnitedStates",
        "Vanuatu",
        "Venezuela",
        "Vietnam",
        "VirginIslandsUs",
        "WallisAndFutuna",
        "Yemen",
    )

    measurements = ("agb", "hba95", "hmax95")

    def __init__(
        self,
        paths="data",
        crs=None,
        res=None,
        measurement="agb",
        country=all_countries[0],
        transforms=None,
        cache=True,
        checksum=True,
        time_series=False,
    ):
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: CRS to warp to (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres)
                format (defaults to the resolution of the first file found)
            measurement: which of the three measurements, 'agb', 'hba95', or
                'hmax95'
            country: country for which to retrieve data
            transforms: a function/transform that takes an input sample and
                returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)
            time_series: if True, stack data along the time series dimension
                [T, H, W, C]. If False, merge data into a [H, W, C] mosaic.

        Raises:
            AssertionError: if country or measurement arg are not str or
                invalid
            DatasetNotFoundError: If dataset is not found.
        """
        self.paths = paths
        self.checksum = checksum

        assert isinstance(country, str), "Country argument must be a str."
        assert country in self.all_countries, (
            f"You have selected an invalid country, please choose one of {self.all_countries}"
        )
        self.country = country

        assert isinstance(measurement, str), "Measurement must be a string."
        assert measurement in self.measurements, (
            f"You have entered an invalid measurement, please choose one of {self.measurements}."
        )
        self.measurement = measurement

        self.filename_glob = f"**/Mangrove_{self.measurement}_{self.country}*"

        self._verify()

        super().__init__(
            paths, crs, res, transforms=transforms, cache=cache, time_series=time_series
        )

    def _verify(self):
        """Verify the integrity of the dataset."""
        if self.files:
            return

        paths = self.paths
        pathname = os.path.join(paths, self.zipfile)
        if os.path.exists(pathname):
            if self.checksum and not check_integrity(pathname, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            self._extract()
            return

        raise DatasetNotFoundError(self)

    def _extract(self):
        """Extract the dataset."""
        paths = self.paths
        pathname = os.path.join(paths, self.zipfile)
        extract_archive(pathname)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        mask = ops.convert_to_numpy(ops.squeeze(sample["mask"]))
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = ops.convert_to_numpy(ops.squeeze(sample["prediction"]))
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
