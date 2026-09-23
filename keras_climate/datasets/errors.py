"""Dataset-specific exceptions (ported from torchgeo.datasets.errors)."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset is requested but doesn't exist."""

    def __init__(self, dataset):
        msg = "Dataset not found"

        if hasattr(dataset, "root"):
            var = "root"
            val = dataset.root
        elif hasattr(dataset, "paths"):
            var = "paths"
            val = dataset.paths
        else:
            super().__init__(f"{msg}.")
            return

        msg += f" in `{var}={val!r}` and "

        if hasattr(dataset, "download") and not dataset.download:
            msg += "`download=False`"
        else:
            msg += "cannot be automatically downloaded"

        msg += f", either specify a different `{var}` or "

        if hasattr(dataset, "download") and not dataset.download:
            msg += "use `download=True` to automatically"
        else:
            msg += "manually"

        msg += " download the dataset."

        super().__init__(msg)


class DependencyNotFoundError(Exception):
    """Raised when an optional dataset dependency is not installed."""


class RGBBandsMissingError(ValueError):
    """Raised when a dataset is missing RGB bands for plotting."""

    def __init__(self):
        super().__init__("Dataset does not contain some of the RGB bands")
