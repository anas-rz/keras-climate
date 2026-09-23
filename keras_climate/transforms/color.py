"""Color transforms (ported from torchgeo.transforms.color)."""

from keras import ops

from ._random import bernoulli_mask, ensure_batched


class RandomGrayscale:
    r"""Apply random transformation to grayscale according to a probability p value.

    There is no single agreed upon definition of grayscale for MSI. Some
    possibilities include:

    * Average of all bands: :math:`\frac{1}{C}` where :math:`C` is the
      number of spectral channels.
    * RGB-only bands: :math:`[0.299, 0.587, 0.114]` for the RGB channels, 0
      for all other channels.
    * PCA: the first principal component across the spectral axis.

    The weight vector you provide will be automatically rescaled to sum to
    1 in order to avoid changing the intensity of the image.

    Expects channels-last (``... x H x W x C``) input.
    """

    def __init__(self, weights, p=0.1, same_on_batch=False):
        """Initialize a new RandomGrayscale instance.

        Args:
            weights: Weights applied to each channel to compute a grayscale
                representation. Should be the same length as the number of
                channels.
            p: Probability of the image to be transformed to grayscale.
            same_on_batch: Apply the same transformation across the batch.
        """
        weights = ops.convert_to_tensor(weights)
        # Rescale to sum to 1
        self.weights = weights / ops.sum(weights)
        self.p = p
        self.same_on_batch = same_on_batch

    def __call__(self, x, training=True):
        x, was_unbatched = ensure_batched(x)
        if not training:
            return ops.squeeze(x, axis=0) if was_unbatched else x

        batch_size = x.shape[0]
        gray = ops.sum(x * self.weights, axis=-1, keepdims=True)
        gray = ops.repeat(gray, x.shape[-1], axis=-1)

        mask = bernoulli_mask(batch_size, self.p, self.same_on_batch)
        mask = ops.reshape(mask, (batch_size,) + (1,) * (len(x.shape) - 1))
        out = ops.where(mask > 0, gray, x)

        return ops.squeeze(out, axis=0) if was_unbatched else out
