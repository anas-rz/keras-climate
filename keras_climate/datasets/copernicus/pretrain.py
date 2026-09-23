"""Copernicus-Pretrain dataset (ported from torchgeo.datasets.copernicus.pretrain)."""

import os
import random

import numpy as np
from keras import ops

from ..utils import lazy_import, quantile_normalization


class CopernicusPretrain:
    """Copernicus-Pretrain dataset.

    Copernicus-Pretrain is an extension of the SSL4EO-S12 dataset to all
    major Sentinel missions (S1-S5P). The images are organized into ~310K
    regional grids (0.25x0.25 degrees, consistent with ERA5), densely
    covering the whole land surface and near-land ocean with time series
    from eight distinct Sentinel modalities.

    This dataset class uses WebDataset for efficient data loading in
    distributed environments, which returns a Python iterable that streams
    samples on demand.

    The full dataset has a varying number of modalities, S1/2 local
    patches, and timestamps for different grids. It also contains metadata
    including the filenames all images are derived from. For simplicity,
    the current dataset class provides a minimum example:

    - only use grids with all modalities (220k)
    - sample one local patch for S1 and S2
    - sample one timestamp for each modality

    Therefore, each sample contains 8 tensors (S1, S2, S3, S5P NO2/CO/SO2/O3, DEM).

    Example:

    .. code-block:: python

       dataset = CopernicusPretrain(
           urls='data/example-{000000..000009}.tar', shardshuffle=True, resampled=True
       )

       # Check the first sample
       sample = next(iter(dataset))
       s1 = sample['s1_grd.pth']
       s2 = sample['s2_toa.pth']
       s3 = sample['s3_olci.pth']
       s5p_co = sample['s5p_co.pth']
       s5p_no2 = sample['s5p_no2.pth']
       s5p_o3 = sample['s5p_o3.pth']
       s5p_so2 = sample['s5p_so2.pth']
       dem = sample['dem.pth']

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849

    .. note::

       This dataset requires the following additional library to be
       installed:

       * `<https://pypi.org/project/webdataset/>`_ to load the dataset.
    """

    url_dict = {
        # grids with all modalities
        "220k_aligned": "https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/ssl4eo_s_220k_aligned/example-{000000..002255}.tar",
        # remaining grids (with at least one modality)
        "220k_310k_union": "https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/ssl4eo_s_220k_310k_union/example-{002256..003210}.tar",
        # 100 example grids
        "100_example": "https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/example_100_grids/example_100_webdataset/example-{000000..000009}.tar",
    }

    def __init__(self, *args, **kwargs):
        """Initialize a new CopernicusPretrain instance.

        Args:
            *args: Arguments passed to the WebDataset base class.
            **kwargs: Keyword arguments passed to the WebDataset base class.
        """
        # https://github.com/torchgeo/torchgeo/security/advisories/GHSA-6gm9-8jxc-p862
        os.environ["WDS_PYTORCH_WEIGHTS_ONLY"] = "1"

        wds = lazy_import("webdataset")

        self.dataset = (
            wds.WebDataset(*args, **kwargs)
            .shuffle(10)  # shuffle individual samples before batching
            .decode()  # decode binary data
            .map(self._drop_metadata)  # remove non-tensor metadata
            .select(self._has_all_modalities)  # select samples with all modalities
            .map(self._sample_one_local_patch)  # sample one local patch for S1 and S2
            .map(self._sample_one_time_stamp)  # sample one timestamp for all modalities
            .map(self._to_tensors)  # convert to Keras tensors
        )

    def __iter__(self):
        """Iterate over images in the dataset.

        Yields:
            sample of images
        """
        return iter(self.dataset)

    def _drop_metadata(self, sample):
        """Mapping function: remove all non-array metadata.

        Args:
            sample: A single sample from the dataset.

        Returns:
            The same sample with only array-like (tensor) entries.
        """
        new_sample = {}
        for key, value in sample.items():
            if hasattr(value, "shape"):
                new_sample[key] = value

        return new_sample

    def _has_all_modalities(self, sample):
        """Selection function: filter samples with all required modalities.

        Args:
            sample: A single sample from the dataset.

        Returns:
            True if all modalities are present in the sample, else False.
        """
        required_keys = [
            "s1_grd.pth",
            "s2_toa.pth",
            "s3_olci.pth",
            "s5p_co.pth",
            "s5p_no2.pth",
            "s5p_o3.pth",
            "s5p_so2.pth",
            "dem.pth",
        ]
        return all(key in sample for key in required_keys)

    def _sample_one_local_patch(self, sample):
        """Mapping function: randomly select one local patch for S1 and S2.

        Args:
            sample: A single sample from the dataset.

        Returns:
            The same sample with only a single patch for S1 and S2.
        """
        s1, s2 = sample["s1_grd.pth"], sample["s2_toa.pth"]
        idx = random.randint(0, s1.shape[0] - 1)
        sample["s1_grd.pth"], sample["s2_toa.pth"] = s1[idx], s2[idx]
        return sample

    def _sample_one_time_stamp(self, sample):
        """Mapping function: randomly select one timestamp for all modalities.

        Args:
            sample: A single sample from the dataset.

        Returns:
            The same sample with only a single timestamp.
        """
        for key in sample:
            if key.endswith(".pth") and key != "dem.pth":
                idx = random.randint(0, sample[key].shape[0] - 1)
                sample[key] = sample[key][idx]

        return sample

    def _to_tensors(self, sample):
        """Mapping function: convert all entries to Keras tensors.

        Args:
            sample: A single sample from the dataset.

        Returns:
            The same sample with all values converted to Keras tensors.
        """
        return {key: ops.convert_to_tensor(np.asarray(value)) for key, value in sample.items()}

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__iter__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=2, ncols=4)

        # Channels-first (C, H, W) samples -> channels-last for display.
        image = sample["s1_grd.pth"]
        vv = image[0]
        vh = image[1]
        image = ops.stack([vv, vh, (vv + vh) / 2], axis=-1)
        image = quantile_normalization(image)
        ax[0, 0].imshow(ops.convert_to_numpy(image))
        ax[0, 0].axis("off")

        rgb_bands = [3, 2, 1]
        image = ops.take(sample["s2_toa.pth"], ops.convert_to_tensor(rgb_bands), axis=0)
        image = ops.cast(image, "float32")
        image = ops.transpose(image, (1, 2, 0))
        image = quantile_normalization(image)
        ax[0, 1].imshow(ops.convert_to_numpy(image))
        ax[0, 1].axis("off")

        rgb_bands = [7, 5, 3]
        image = ops.take(sample["s3_olci.pth"], ops.convert_to_tensor(rgb_bands), axis=0)
        image = ops.transpose(image, (1, 2, 0))
        image = quantile_normalization(image)
        ax[0, 2].imshow(ops.convert_to_numpy(image))
        ax[0, 2].axis("off")

        image = ops.convert_to_numpy(sample["dem.pth"])
        ax[0, 3].imshow(image, cmap="terrain")
        ax[0, 3].axis("off")

        image = ops.convert_to_numpy(sample["s5p_co.pth"])[0]
        ax[1, 0].imshow(image, cmap="Wistia")
        ax[1, 0].axis("off")

        image = ops.convert_to_numpy(sample["s5p_no2.pth"])[0]
        ax[1, 1].imshow(image, cmap="Wistia")
        ax[1, 1].axis("off")

        image = ops.convert_to_numpy(sample["s5p_o3.pth"])[0]
        ax[1, 2].imshow(image, cmap="Wistia")
        ax[1, 2].axis("off")

        image = ops.convert_to_numpy(sample["s5p_so2.pth"])[0]
        ax[1, 3].imshow(image, cmap="Wistia")
        ax[1, 3].axis("off")

        if show_titles:
            ax[0, 0].set_title("S1 GRD")
            ax[0, 1].set_title("S2 TOA")
            ax[0, 2].set_title("S3 OLCI")
            ax[0, 3].set_title("DEM")
            ax[1, 0].set_title("S5P CO")
            ax[1, 1].set_title("S5P NO$_2$")
            ax[1, 2].set_title("S5P O$_3$")
            ax[1, 3].set_title("S5P SO$_2$")

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()

        return fig
