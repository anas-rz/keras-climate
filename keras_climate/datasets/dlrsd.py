"""DLRSD dataset (ported from torchgeo.datasets.dlrsd)."""

import glob
import os

import numpy as np
import pandas as pd
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_url, draw_semantic_segmentation_masks, extract_archive


class DLRSDBase(NonGeoDataset):
    """Shared base for DLRSD and DLRSDMultilabel.

    The `DLRSD <https://sites.google.com/view/zhouwx/dataset>`__
    dataset is a dataset for dense labeling of remote sensing imagery. It
    contains 2100 images of size 256x256 pixels across 21 scene categories
    (100 images per class), and is derived from the UC Merced Land Use
    Dataset.

    Dataset features:

    * 2100 images with 0.3m spatial resolution (256x256 px)
    * three spectral bands - RGB

    Dataset classes:

    0. airplane
    1. bare soil
    2. buildings
    3. cars
    4. chaparral
    5. court
    6. dock
    7. field
    8. grass
    9. mobile home
    10. pavement
    11. sand
    12. sea
    13. ship
    14. tanks
    15. trees
    16. water

    This base class provides shared dataset assets and download/extract/verify
    plumbing for both the semantic segmentation and multi-label variants of
    DLRSD.
    """

    url = "https://hf.co/datasets/calebrob6/dlrsd/resolve/dcd622dd05f327cd0fd10951ad6cd7ba52d0e832/"

    filename = "DLRSD.zip"
    sha256 = "cb49d850855a7622c51b5f09dbdf98952e064b661f35cb105e8749513e75796f"

    directory = "DLRSD"

    image_fns = []

    classes = (
        "airplane",
        "bare soil",
        "buildings",
        "cars",
        "chaparral",
        "court",
        "dock",
        "field",
        "grass",
        "mobile home",
        "pavement",
        "sand",
        "sea",
        "ship",
        "tanks",
        "trees",
        "water",
    )

    def __init__(self, root="data", transforms=None, download=False, checksum=True):
        """Initialize a new DLRSD-style dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the SHA256 of the downloaded files (may
                be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

    def __len__(self):
        return len(self.image_fns)

    def _load_image(self, index):
        """Load a single image, returned channels-last (H, W, C)."""
        path = self.image_fns[index]
        with Image.open(path) as img:
            array = np.array(img.convert("RGB")).astype("float32")
            return ops.convert_to_tensor(array)

    def _verify(self):
        """Verify the integrity of the dataset.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
            RuntimeError: If an existing archive is corrupted.
        """
        filepath = os.path.join(self.root, self.directory)
        if os.path.exists(filepath):
            return

        filepath = os.path.join(self.root, self.filename)
        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, sha256=self.sha256):
                raise RuntimeError("Dataset found, but corrupted.")
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self):
        """Download the dataset."""
        # Empty sha256 is treated as "no checksum" by check_integrity.
        download_url(self.url + self.filename, self.root, sha256=self.sha256 if self.checksum else "")

    def _extract(self):
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)


class DLRSD(DLRSDBase):
    """DLRSD semantic segmentation dataset.

    The :class:`DLRSD` variant provides pixel-level semantic segmentation
    annotations for 17 land cover classes.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.3390/rs10060964
    """

    colormap = (
        (166, 202, 240),
        (128, 128, 0),
        (0, 0, 128),
        (255, 0, 0),
        (0, 128, 0),
        (128, 0, 0),
        (255, 233, 233),
        (160, 160, 164),
        (0, 128, 128),
        (90, 87, 255),
        (255, 255, 0),
        (255, 192, 0),
        (0, 0, 255),
        (255, 0, 192),
        (128, 0, 128),
        (0, 255, 0),
        (0, 255, 255),
    )

    def __init__(self, root="data", transforms=None, download=False, checksum=True):
        """Initialize a new DLRSD dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the SHA256 of the downloaded files (may
                be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        super().__init__(root, transforms, download, checksum)

        images_dir = os.path.join(root, self.directory, "Images")
        labels_dir = os.path.join(root, self.directory, "Labels")
        self.image_fns = sorted(glob.glob(os.path.join(images_dir, "*", "*.tif")))
        self.mask_fns = [
            os.path.join(labels_dir, os.path.basename(os.path.dirname(p)), os.path.basename(p).replace(".tif", ".png"))
            for p in self.image_fns
        ]

    def __getitem__(self, index):
        image = self._load_image(index)
        mask = self._load_mask(index)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_mask(self, index):
        """Load a single mask."""
        path = self.mask_fns[index]
        with Image.open(path) as img:
            array = np.array(img).astype("int64")
            # Palette indices are 1-17, convert to 0-indexed
            return ops.convert_to_tensor(array - 1)

    def plot(self, sample, show_titles=True, suptitle=None, alpha=0.5):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        if image.max() > 1:
            image = image.astype("float32") / 255.0

        # Use a uint8 0-255 copy of the image for the segmentation overlay so
        # that user transforms which normalize the tensor to [0, 1] don't
        # produce a near-black overlay (draw_semantic_segmentation_masks
        # casts to uint8).
        overlay_image = sample["image"]
        overlay_np = ops.convert_to_numpy(overlay_image)
        if np.issubdtype(overlay_np.dtype, np.floating):
            if overlay_np.max() <= 1:
                overlay_image = ops.cast(overlay_image * 255, "uint8")
            else:
                overlay_image = ops.cast(overlay_image, "uint8")

        mask_overlay = draw_semantic_segmentation_masks(
            overlay_image, sample["mask"], alpha=alpha, colors=list(self.colormap)
        )

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred_overlay = draw_semantic_segmentation_masks(
                overlay_image, sample["prediction"], alpha=alpha, colors=list(self.colormap)
            )

        ncols = 3 if showing_predictions else 2
        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask_overlay)
        axs[1].axis("off")
        if showing_predictions:
            axs[2].imshow(pred_overlay)
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Ground Truth")
            if showing_predictions:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class DLRSDMultilabel(DLRSDBase):
    """DLRSD multi-label scene classification dataset.

    The :class:`DLRSDMultilabel` variant provides multi-label scene
    classification annotations with 17 label classes.

    If you use this dataset in your research, please cite the following papers:

    * https://doi.org/10.3390/rs10060964
    * https://doi.org/10.1109/TGRS.2017.2760909
    """

    def __init__(self, root="data", transforms=None, download=False, checksum=True):
        """Initialize a new DLRSDMultilabel dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the SHA256 of the downloaded files (may
                be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        super().__init__(root, transforms, download, checksum)

        images_dir = os.path.join(root, self.directory, "Images")
        self.image_fns = sorted(glob.glob(os.path.join(images_dir, "*", "*.tif")))

        self.multilabels = {}
        csv_path = os.path.join(root, self.directory, "multilabels.csv")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            self.multilabels[row.iloc[0]] = [int(x) for x in row.iloc[1:]]

    def __getitem__(self, index):
        image = self._load_image(index)
        label = self._load_target(index)
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_target(self, index):
        """Load multi-label target for a single image."""
        stem = os.path.splitext(os.path.basename(self.image_fns[index]))[0]
        labels = self.multilabels[stem]
        return ops.convert_to_tensor(np.array(labels, dtype="int64"))

    def _onehot_labels_to_names(self, label_mask):
        """Get a list of class names given a label mask."""
        return [self.classes[i] for i, v in enumerate(label_mask) if v]

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        if image.max() > 1:
            image = image.astype("float32") / 255.0

        label_mask = ops.convert_to_numpy(sample["label"]).astype(bool)
        labels = self._onehot_labels_to_names(label_mask)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = ops.convert_to_numpy(sample["prediction"]).astype(bool)
            predictions = self._onehot_labels_to_names(prediction_mask)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Labels: {', '.join(labels)}"
            if showing_predictions:
                title += f"\nPredictions: {', '.join(predictions)}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
