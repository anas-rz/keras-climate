"""Mixins for dataset classes (ported from torchgeo.datasets.mixins)."""


class PlottingMixin:
    """Mixin for dataset plotting."""

    #: Names of all available bands in the dataset
    all_bands: tuple = ()

    #: Names of RGB bands in the dataset
    rgb_bands: tuple = ()

    #: Color map for the dataset
    cmap = None
