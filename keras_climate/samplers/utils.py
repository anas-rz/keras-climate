"""Common sampler utilities (ported from torchgeo.samplers.utils)."""

import math

import numpy as np


def _to_tuple(value):
    """Convert value to a tuple if it is not already a tuple."""
    if isinstance(value, (int, float)):
        return (value, value)
    else:
        return value


def get_random_bounding_box(bounds, size, res, generator=None):
    """Returns a random bounding box within a given bounding box.

    .. deprecated:: Use geopandas.GeoSeries.sample_points instead.

    The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the
          height and width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is
          used for the height dimension, and the second *float* for the
          width dimension

    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample
        res: the resolution of the image
        generator: pseudo-random number generator (PRNG), a
            :class:`numpy.random.Generator` or seed.

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    rng = generator if isinstance(generator, np.random.Generator) else np.random.default_rng(generator)

    xmin, ymin, xmax, ymax = bounds
    t_size = _to_tuple(size)
    t_res = _to_tuple(res)

    # May be negative if bounding box is smaller than patch size
    width = (xmax - xmin - t_size[1]) / t_res[0]
    height = (ymax - ymin - t_size[0]) / t_res[1]

    # Use an integer multiple of res to avoid resampling
    xmin += int(rng.random() * width) * t_res[0]
    ymin += int(rng.random() * height) * t_res[1]

    xmax = xmin + t_size[1]
    ymax = ymin + t_size[0]

    return slice(xmin, xmax), slice(ymin, ymax)


def tile_to_chips(bounds, size, stride=None):
    r"""Compute number of chips that can be sampled from a tile.

    Let :math:`i` be the size of the input tile. Let :math:`k` be the
    requested size of the output patch. Let :math:`s` be the requested
    stride. Let :math:`o` be the number of output chips sampled from each
    tile:

        o = ceil((i - k) / s) + 1

    We use ceiling instead of floor to include the final remaining chip in
    each row/column when bounds is not an integer multiple of stride.

    .. deprecated:: Use :func:`convolution_arithmetic` instead.

    Args:
        bounds: bounding box of tile
        size: size of output patch
        stride: stride with which to sample (defaults to ``size``)

    Returns:
        the number of rows/columns that can be sampled
    """
    if stride is None:
        stride = size

    assert stride[0] > 0
    assert stride[1] > 0

    xmin, ymin, xmax, ymax = bounds

    rows = math.ceil((ymax - ymin - size[0]) / stride[0]) + 1
    cols = math.ceil((xmax - xmin - size[1]) / stride[1]) + 1

    return rows, cols


def convolution_arithmetic(input_size, kernel_size, stride=None):
    r"""Compute number of spatial/temporal windows that can be sampled via convolution.

    Let :math:`i` be the size of the input window.
    Let :math:`k` be the requested size of the output window.
    Let :math:`s` be the requested stride.
    Let :math:`o` be the number of output windows sampled from each input:

        o = ceil((i - k) / s) + 1

    We use ceiling instead of floor to include the final remaining window in
    each input when *input_size* is not an integer multiple of *stride*.

    Args:
        input_size: Size of the input window.
        kernel_size: Size of each output window.
        stride: Stride with which to sample (defaults to *input_size*).

    Returns:
        The number of output windows that can be sampled.
    """
    stride = stride or kernel_size
    return math.ceil((input_size - kernel_size) / stride) + 1


def prism(x, y, z):
    """Convert x, y, z coordinates to the vertices of a prism.

    Args:
        x: All x coordinates of a Polygon.
        y: All y coordinates of a Polygon.
        z: Two z coordinates to project the Polygon into.

    Returns:
        The vertices of a 3D prism.
    """
    assert len(x) == len(y)
    assert len(z) == 2
    verts = []

    z0 = z[0].repeat(len(x))
    verts.append(np.stack([x, y, z0]).T)

    z1 = z[1].repeat(len(x))
    verts.append(np.stack([x, y, z1]).T)

    zi = np.array([z[0], z[0], z[1], z[1], z[0]])
    for i in range(len(x) - 1):
        xi = np.array([x[i], x[i + 1], x[i + 1], x[i], x[i]])
        yi = np.array([y[i], y[i + 1], y[i + 1], y[i], y[i]])
        verts.append(np.stack([xi, yi, zi]).T)

    return verts
