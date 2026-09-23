"""Spatial augmentations (ported from torchgeo.transforms.spatial)."""

import numpy as np
import keras
from keras import ops

from ._random import ensure_batched


class SatSlideMix:
    """Applies the Sat-SlideMix augmentation to a batch of images and masks.

    Sat-SlideMix rolls (circularly shifts) images along either the height
    or width axis by a random amount.

    If you use this method in your research, please cite:
    https://doi.org/10.1609/aaai.v39i27.35028

    Expects channels-last (``... x H x W x C``) input.
    """

    def __init__(self, gamma=1, beta=(0.0, 1.0), p=0.5):
        """Initialize a new SatSlideMix instance.

        Args:
            gamma: The number of augmented samples to create for each input
                image. The output batch size will be ``gamma * B``.
            beta: The range of percentage (0.0 to 1.0) of the image
                dimension (height or width) to shift.
            p: Probability to apply the augmentation on each sample.

        Raises:
            AssertionError: If ``gamma`` is not a positive integer.
        """
        if not (isinstance(gamma, int) and gamma > 0):
            raise AssertionError("gamma must be a positive integer")
        self.gamma = gamma
        self.beta = tuple(beta) if isinstance(beta, (tuple, list)) else (0.0, float(beta))
        self.p = p

    def __call__(self, x, training=True):
        x, was_unbatched = ensure_batched(x)
        # Squeezing back to unbatched only makes sense if gamma didn't grow
        # the batch dimension beyond size 1.
        squeeze = was_unbatched and self.gamma == 1

        out = ops.convert_to_numpy(ops.repeat(x, self.gamma, axis=0))
        if not training:
            out_t = ops.convert_to_tensor(out)
            return ops.squeeze(out_t, axis=0) if squeeze else out_t

        n, height, width = out.shape[0], out.shape[1], out.shape[2]
        beta_lo, beta_hi = self.beta

        rng_beta = ops.convert_to_numpy(
            keras.random.uniform((n,), minval=beta_lo, maxval=beta_hi)
        )
        rng_axis = ops.convert_to_numpy(keras.random.uniform((n,)))
        rng_dir = ops.convert_to_numpy(keras.random.uniform((n,)))
        rng_apply = ops.convert_to_numpy(keras.random.uniform((n,)))

        for i in range(n):
            if rng_apply[i] >= self.p:
                continue
            axis = 0 if rng_axis[i] < 0.5 else 1
            direction = 1 if rng_dir[i] < 0.5 else -1
            size = height if axis == 0 else width
            shift = int(round(float(rng_beta[i]) * size)) * direction
            out[i] = np.roll(out[i], shift=shift, axis=axis)

        out_t = ops.convert_to_tensor(out)
        return ops.squeeze(out_t, axis=0) if squeeze else out_t
