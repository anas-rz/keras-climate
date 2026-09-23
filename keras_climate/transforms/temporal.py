"""Temporal transforms (ported from torchgeo.transforms.temporal)."""

from einops import rearrange


class Rearrange:
    """Rearrange tensor dimensions.

    Examples:
        To insert a time dimension::

            Rearrange('b (t c) h w -> b t c h w', c=1)

        To collapse the time dimension::

            Rearrange('b t c h w -> b (t c) h w')
    """

    def __init__(self, *args, **kwargs):
        """Initialize a Rearrange instance.

        Args:
            *args: Positional arguments for :func:`einops.rearrange`.
            **kwargs: Keyword arguments for :func:`einops.rearrange`.
        """
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return rearrange(x, *self.args, **self.kwargs)
