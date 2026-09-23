"""Embedded Seamless Data (ported from torchgeo.datasets.esd)."""

import numpy as np
from keras import ops

from .geo import RasterDataset


class ESDQuantizer:
    """Decode ESD-encoded quantized indices into continuous embedding vectors.

    The ESDQuantizer converts integer quantization indices produced by an
    ESD quantizer into continuous values in the range [-1, 1], representing
    multi-level embeddings of the original input. This enables downstream
    tasks, such as visualization, machine learning, or spatial analysis, to
    operate directly on decoded embeddings without reconstructing the full
    raw data.

    Key points:

    * Factorized decoding: Each index is split into multiple levels according
      to the quantizer configuration.
    * Continuous mapping: Level indices are rescaled and centered to [-1, 1],
      preserving relative distances in embedding space.
    * Fully vectorized: The decoding is performed on entire tensors at once.

    .. note::
       Unlike torchgeo (which moves the decoded level axis to a
       channels-first position), this port keeps the last axis as the
       channel axis (matching keras_climate's channels-last convention) and
       instead moves the raster-band axis (which behaves like a "time"
       dimension of factorized codes) to the front, producing a
       ``(T, H, W, L)`` tensor, matching the ``time_series`` layout already
       used by :class:`~keras_climate.datasets.geo.RasterDataset`.
    """

    def __init__(self, levels=(8, 8, 8, 5, 5, 5)):
        """Initialize the quantization levels for the embedding.

        Args:
            levels: sequence of integers specifying the number of
                quantization levels for each embedding dimension.
        """
        self._levels = np.array(levels, dtype="int32")
        self._basis = np.cumprod(np.array([1, *levels[:-1]], dtype="int32"))

    def indices_to_codes(self, indices):
        """Convert embedding indices to normalized continuous codes.

        Args:
            indices: a tensor of integer indices representing quantized
                embeddings. Shape can be arbitrary (...,).

        Returns:
            Tensor of shape ``(..., len(levels))`` with values normalized to
            [-1, 1].
        """
        indices = ops.cast(ops.convert_to_tensor(indices), "int32")
        indices = ops.expand_dims(indices, axis=-1)

        basis = ops.convert_to_tensor(self._basis)
        levels = ops.convert_to_tensor(self._levels)
        level_indices = ops.mod(ops.floor_divide(indices, basis), levels)

        half = ops.cast(self._levels // 2, "float32")
        codes = (ops.cast(level_indices, "float32") - half) / half

        return codes

    def quantize(self, input):
        """Quantize the input tensor using predefined levels.

        Args:
            input: a tensor containing integer embedding indices of shape
                ``(H, W, T)`` (channels-last raster bands).

        Returns:
            Tensor of quantized codes with normalized values in [-1, 1] and
            shape ``(T, H, W, L)``, where ``L`` is the number of levels.
        """
        codes = self.indices_to_codes(input)  # (H, W, T, L)
        return ops.moveaxis(codes, -2, 0)  # (T, H, W, L)


class EmbeddedSeamlessData(RasterDataset):
    """Embedded Seamless Data (ESD).

    The `Embedded Seamless Data (ESD) <https://arxiv.org/abs/2601.11183>`__
    is a global, analysis-ready Earth embedding dataset at 30-meter
    resolution, designed to overcome the computational and storage
    challenges of planetary-scale Earth system science. By transforming
    multi-sensor satellite observations into compact, quantized latent
    vectors, ESD reduces the original data volume (~1 PB for a full year of
    global land surfaces) to approximately 2.4 TB, enabling decadal-scale
    analysis on standard workstations.

    Key features:

    * **Longitudinal Consistency**: Provides a continuous record from 2000
      to 2024, harmonized from Landsat 5, 7, 8, 9, MODIS Terra and NASADEM
      imagery.
    * **High Reconstructive Fidelity**: Achieves a Mean Absolute Error (MAE)
      of 0.013 across six spectral bands.
    * **Semantic Intelligence**: Captures complex land surface patterns.
    * **Implicit Denoising**: Filters transient noise such as clouds and
      shadows via the ESDNet architecture.
    * **Few-Shot Proficiency**: Supports robust learning with minimal
      labeled data.

    If you use this dataset in your research, please refer to:

    * Paper: https://arxiv.org/abs/2601.11183
    * Code: https://github.com/shuangchencc/ESD
    * Dataset: https://data-starcloud.pcl.ac.cn/iearthdata/64
    """

    # SDC30_EBD_V001_02VMN_2024.tif
    filename_glob = "SDC30_EBD_*"
    filename_regex = r".*_(?P<date>\d{4})"
    date_format = "%Y"

    quantizer = ESDQuantizer()

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["image"] = self.quantizer.quantize(sample["image"])
        return sample

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        vectors = ops.convert_to_numpy(sample["image"])  # (T, H, W, L)
        _months, H, W, _channels = vectors.shape

        # Compute valid mask: any non-zero pixel across channels, combining
        # the first 12 "months" (raster bands).
        valid_mask = ~np.isclose(vectors, 0.0, atol=1e-6)
        valid_mask = valid_mask[:12].any(axis=-1).any(axis=0)

        # Reduce channels to RGB using mean over selected levels.
        R = (vectors[:, :, :, 5].mean(axis=0) + 1) / 2
        G = (vectors[:, :, :, 1].mean(axis=0) + 1) / 2
        B = (vectors[:, :, :, 2].mean(axis=0) + 1) / 2

        disp_img = np.zeros((H, W, 4), dtype=np.uint8)
        disp_img[..., 0] = (np.clip(R, 0, 1) * 255).astype(np.uint8)
        disp_img[..., 1] = (np.clip(G, 0, 1) * 255).astype(np.uint8)
        disp_img[..., 2] = (np.clip(B, 0, 1) * 255).astype(np.uint8)
        disp_img[..., 3] = valid_mask.astype(np.uint8) * 255

        fig, ax = plt.subplots()
        ax.imshow(disp_img)
        ax.axis("off")

        if show_titles:
            ax.set_title("ESD Embedding Visualization")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
