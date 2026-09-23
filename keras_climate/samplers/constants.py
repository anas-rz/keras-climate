"""Common sampler constants (ported from torchgeo.samplers.constants)."""

from enum import Enum, auto


class Units(Enum):
    """Enumeration defining units of ``size`` parameter.

    Used by :class:`~keras_climate.samplers.SpatialSampler`.
    """

    #: Units in number of pixels
    PIXELS = auto()

    #: Units of coordinate reference system (CRS)
    CRS = auto()
