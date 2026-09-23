"""SAR-specific transforms for synthetic aperture radar imagery.

Ported from torchgeo.transforms.sar.
"""

from keras import ops

from ._random import bernoulli_mask, ensure_batched


def _box_filter(x, window_size):
    """Apply a per-channel box (mean) filter over the spatial dimensions.

    Args:
        x: Input tensor of shape ``(B, H, W, C)``.
        window_size: Odd integer side length of the smoothing window.

    Returns:
        Smoothed tensor of identical shape.
    """
    pad = window_size // 2
    x_padded = ops.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="reflect")
    return ops.average_pool(
        x_padded,
        pool_size=(window_size, window_size),
        strides=(1, 1),
        padding="valid",
        data_format="channels_last",
    )


def lee_filter(image, window_size=7, num_looks=1.0, eps=1e-8):
    r"""Apply the Lee filter to a SAR intensity image.

    The Lee (1980) filter assumes a multiplicative speckle model
    :math:`x = s \cdot v` where :math:`s` is the underlying signal and
    :math:`v` is unit-mean speckle with variance
    :math:`\sigma_v^2 = 1 / L` for an :math:`L`-look intensity image. The
    local linear minimum mean square error (LMMSE) estimator is

    .. math::

        \hat{s} = \mu + k \cdot (x - \mu),
        \quad
        k = \frac{\sigma_s^2}{\sigma_s^2 + \sigma_v^2 \mu^2}

    In homogeneous regions the filter behaves like a mean filter; near edges
    it preserves detail by giving the local mean less weight.

    If you use this method in your research, please cite:
    https://doi.org/10.1109/TPAMI.1980.4766994

    Args:
        image: SAR intensity tensor of shape ``(B, H, W, C)`` (or ``(H, W, C)``).
            Values are assumed non-negative intensities, not amplitudes or dB.
        window_size: Odd integer size of the local statistics window. Larger
            values produce more smoothing at the cost of detail.
        num_looks: Equivalent number of looks (ENL) of the input image.
            Single-look complex (SLC) intensity has ``num_looks=1``.
            Sentinel-1 GRDH typically has ``num_looks`` near 5.
        eps: Numerical floor to avoid division by zero in flat regions.

    Returns:
        Filtered tensor of the same shape and dtype as ``image``.

    Raises:
        ValueError: If ``window_size`` is not a positive odd integer.
        ValueError: If ``num_looks`` is not strictly positive.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    if num_looks <= 0:
        raise ValueError(f"num_looks must be > 0, got {num_looks}")

    image, was_unbatched = ensure_batched(image)

    sigma_v_sq = 1.0 / float(num_looks)

    mean_local = _box_filter(image, window_size)
    mean_sq_local = _box_filter(image * image, window_size)
    var_local = ops.maximum(mean_sq_local - mean_local * mean_local, 0.0)

    var_signal = ops.maximum(var_local - sigma_v_sq * mean_local * mean_local, 0.0)
    weight = var_signal / (var_signal + sigma_v_sq * mean_local * mean_local + eps)

    out = mean_local + weight * (image - mean_local)
    return ops.squeeze(out, axis=0) if was_unbatched else out


class LeeFilter:
    """Lee speckle reduction filter for SAR imagery.

    Applies the classic Lee (1980) adaptive filter to reduce multiplicative
    speckle noise while preserving edges and structural detail. Operates on
    SAR intensity imagery with non-negative values; amplitude and dB inputs
    should be converted to intensity beforehand.

    If you use this method in your research, please cite:
    https://doi.org/10.1109/TPAMI.1980.4766994
    """

    def __init__(self, window_size=7, num_looks=1.0, p=1.0, same_on_batch=False):
        """Initialize a new LeeFilter instance.

        Args:
            window_size: Odd integer size of the local statistics window.
            num_looks: Equivalent number of looks (ENL) of the input SAR data.
            p: Probability of applying the filter to each sample.
            same_on_batch: Apply the same transformation across the batch.

        Raises:
            ValueError: If ``window_size`` is not a positive odd integer.
            ValueError: If ``num_looks`` is not strictly positive.
        """
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be a positive odd integer, got {window_size}"
            )
        if num_looks <= 0:
            raise ValueError(f"num_looks must be > 0, got {num_looks}")
        self.window_size = window_size
        self.num_looks = num_looks
        self.p = p
        self.same_on_batch = same_on_batch

    def __call__(self, x, training=True):
        x, was_unbatched = ensure_batched(x)
        if not training:
            return ops.squeeze(x, axis=0) if was_unbatched else x

        filtered = lee_filter(x, self.window_size, self.num_looks)

        batch_size = x.shape[0]
        mask = bernoulli_mask(batch_size, self.p, self.same_on_batch)
        mask = ops.reshape(mask, (batch_size,) + (1,) * (len(x.shape) - 1))
        out = ops.where(mask > 0, filtered, x)

        return ops.squeeze(out, axis=0) if was_unbatched else out
