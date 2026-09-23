"""Spectral index transforms (ported from torchgeo.transforms.indices).

For more information about indices see:
- https://www.indexdatabase.de/db/i.php
- https://github.com/awesome-spectral-indices/awesome-spectral-indices

Expects channels-last (``... x H x W x C``) input; each index is appended
as an additional trailing channel.
"""

from keras import ops

_EPSILON = 1e-10


class AppendNormalizedDifferenceIndex:
    r"""Append normalized difference index as channel to image tensor.

    Computes the following index:

    .. math::

       \text{NDI} = \frac{A - B}{A + B}
    """

    def __init__(self, index_a, index_b):
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index
        """
        self.index_a = index_a
        self.index_b = index_b

    def __call__(self, x):
        x = ops.convert_to_tensor(x)
        band_a = x[..., self.index_a]
        band_b = x[..., self.index_b]
        ndi = (band_a - band_b) / (band_a + band_b + _EPSILON)
        ndi = ops.expand_dims(ndi, -1)
        return ops.concatenate((x, ndi), axis=-1)


class AppendNBR(AppendNormalizedDifferenceIndex):
    r"""Normalized Burn Ratio (NBR).

    .. math::

       \text{NBR} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}

    https://www.yumpu.com/en/document/view/24226870/the-normalized-burn-ratio-and-relationships-to-burn-severity-/7
    """

    def __init__(self, index_nir, index_swir):
        super().__init__(index_a=index_nir, index_b=index_swir)


class AppendNDBI(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Built-up Index (NDBI).

    .. math::

       \text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}

    https://doi.org/10.1080/01431160304987
    """

    def __init__(self, index_swir, index_nir):
        super().__init__(index_a=index_swir, index_b=index_nir)


class AppendNDSI(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Snow Index (NDSI).

    .. math::

       \text{NDSI} = \frac{\text{G} - \text{SWIR}}{\text{G} + \text{SWIR}}

    https://doi.org/10.1109/IGARSS.1994.399618
    """

    def __init__(self, index_green, index_swir):
        super().__init__(index_a=index_green, index_b=index_swir)


class AppendNDVI(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Vegetation Index (NDVI).

    .. math::

       \text{NDVI} = \frac{\text{NIR} - \text{R}}{\text{NIR} + \text{R}}

    https://doi.org/10.1016/0034-4257(79)90013-0
    """

    def __init__(self, index_nir, index_red):
        super().__init__(index_a=index_nir, index_b=index_red)


class AppendNDWI(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Water Index (NDWI).

    .. math::

       \text{NDWI} = \frac{\text{G} - \text{NIR}}{\text{G} + \text{NIR}}

    https://doi.org/10.1080/01431169608948714
    """

    def __init__(self, index_green, index_nir):
        super().__init__(index_a=index_green, index_b=index_nir)


class AppendMNDWI(AppendNormalizedDifferenceIndex):
    r"""Modified Normalized Difference Water Index (MNDWI).

    .. math::

       \text{MNDWI} = \frac{\text{G} - \text{SWIR}}{\text{G} + \text{SWIR}}

    https://doi.org/10.1080/01431160600589179
    """

    def __init__(self, index_green, index_swir):
        super().__init__(index_a=index_green, index_b=index_swir)


class AppendSWI(AppendNormalizedDifferenceIndex):
    r"""Standardized Water-Level Index (SWI).

    .. math::

       \text{SWI} = \frac{\text{VRE1} - \text{SWIR2}}{\text{VRE1} + \text{SWIR2}}

    https://doi.org/10.3390/w13121647
    """

    def __init__(self, index_vre1, index_swir2):
        super().__init__(index_a=index_vre1, index_b=index_swir2)


class AppendGNDVI(AppendNormalizedDifferenceIndex):
    r"""Green Normalized Difference Vegetation Index (GNDVI).

    .. math::

       \text{GNDVI} = \frac{\text{NIR} - \text{G}}{\text{NIR} + \text{G}}

    https://doi.org/10.2134/agronj2001.933583x
    """

    def __init__(self, index_nir, index_green):
        super().__init__(index_a=index_nir, index_b=index_green)


class AppendBNDVI(AppendNormalizedDifferenceIndex):
    r"""Blue Normalized Difference Vegetation Index (BNDVI).

    .. math::

       \text{BNDVI} = \frac{\text{NIR} - \text{B}}{\text{NIR} + \text{B}}

    https://doi.org/10.1016/S1672-6308(07)60027-4
    """

    def __init__(self, index_nir, index_blue):
        super().__init__(index_a=index_nir, index_b=index_blue)


class AppendNDRE(AppendNormalizedDifferenceIndex):
    r"""Normalized Difference Red Edge Vegetation Index (NDRE).

    .. math::

       \text{NDRE} = \frac{\text{NIR} - \text{VRE1}}{\text{NIR} + \text{VRE1}}

    https://agris.fao.org/agris-search/search.do?recordID=US201300795763
    """

    def __init__(self, index_nir, index_vre1):
        super().__init__(index_a=index_nir, index_b=index_vre1)


class AppendTriBandNormalizedDifferenceIndex:
    r"""Append normalized difference index involving 3 bands as channel to image tensor.

    .. math::

       \text{TBNDI} = \frac{A - (B + C)}{A + (B + C)}
    """

    def __init__(self, index_a, index_b, index_c):
        """Initialize a new transform instance.

        Args:
            index_a: reference band channel index
            index_b: difference band channel index of component 1
            index_c: difference band channel index of component 2
        """
        self.index_a = index_a
        self.index_b = index_b
        self.index_c = index_c

    def __call__(self, x):
        x = ops.convert_to_tensor(x)
        band_a = x[..., self.index_a]
        band_b = x[..., self.index_b]
        band_c = x[..., self.index_c]
        band_d = band_b + band_c
        tbndi = (band_a - band_d) / (band_a + band_d + _EPSILON)
        tbndi = ops.expand_dims(tbndi, -1)
        return ops.concatenate((x, tbndi), axis=-1)


class AppendGRNDVI(AppendTriBandNormalizedDifferenceIndex):
    r"""Green-Red Normalized Difference Vegetation Index (GRNDVI).

    .. math::

       \text{GRNDVI} =
           \frac{\text{NIR} - (\text{G} + \text{R})}{\text{NIR} + (\text{G} + \text{R})}

    https://doi.org/10.1016/S1672-6308(07)60027-4
    """

    def __init__(self, index_nir, index_green, index_red):
        super().__init__(index_a=index_nir, index_b=index_green, index_c=index_red)


class AppendGBNDVI(AppendTriBandNormalizedDifferenceIndex):
    r"""Green-Blue Normalized Difference Vegetation Index (GBNDVI).

    .. math::

       \text{GBNDVI} =
           \frac{\text{NIR} - (\text{G} + \text{B})}{\text{NIR} + (\text{G} + \text{B})}

    https://doi.org/10.1016/S1672-6308(07)60027-4
    """

    def __init__(self, index_nir, index_green, index_blue):
        super().__init__(index_a=index_nir, index_b=index_green, index_c=index_blue)


class AppendRBNDVI(AppendTriBandNormalizedDifferenceIndex):
    r"""Red-Blue Normalized Difference Vegetation Index (RBNDVI).

    .. math::

       \text{RBNDVI} =
           \frac{\text{NIR} - (\text{R} + \text{B})}{\text{NIR} + (\text{R} + \text{B})}

    https://doi.org/10.1016/S1672-6308(07)60027-4
    """

    def __init__(self, index_nir, index_red, index_blue):
        super().__init__(index_a=index_nir, index_b=index_red, index_c=index_blue)


class AppendSAVI:
    r"""Soil-Adjusted Vegetation Index (SAVI).

    .. math::

       \text{SAVI} = \frac{1.5 \times (\text{NIR} - \text{R})}
           {\text{NIR} + \text{R} + 0.5}

    https://doi.org/10.1016/0034-4257(88)90106-X
    """

    def __init__(self, index_nir, index_red):
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_red: index of the Red band in the image
        """
        self.index_nir = index_nir
        self.index_red = index_red

    def __call__(self, x):
        x = ops.convert_to_tensor(x)
        nir = x[..., self.index_nir]
        red = x[..., self.index_red]
        savi = 1.5 * (nir - red) / (nir + red + 0.5 + _EPSILON)
        savi = ops.expand_dims(savi, -1)
        return ops.concatenate((x, savi), axis=-1)


class AppendEVI:
    r"""Enhanced Vegetation Index (EVI).

    .. math::

       \text{EVI} = \frac{2.5 \times (\text{NIR} - \text{R})}
           {\text{NIR} + 6 \times \text{R} - 7.5 \times \text{B} + 1}

    https://doi.org/10.1016/S0034-4257(96)00112-5
    """

    def __init__(self, index_nir, index_red, index_blue):
        """Initialize a new transform instance.

        Args:
            index_nir: index of the Near Infrared (NIR) band in the image
            index_red: index of the Red band in the image
            index_blue: index of the Blue band in the image
        """
        self.index_nir = index_nir
        self.index_red = index_red
        self.index_blue = index_blue

    def __call__(self, x):
        x = ops.convert_to_tensor(x)
        nir = x[..., self.index_nir]
        red = x[..., self.index_red]
        blue = x[..., self.index_blue]
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + _EPSILON)
        evi = ops.expand_dims(evi, -1)
        return ops.concatenate((x, evi), axis=-1)
