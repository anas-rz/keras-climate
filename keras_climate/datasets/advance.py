"""ADVANCE dataset (ported from torchgeo.datasets.advance)."""

import glob
import os

import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive, lazy_import


class ADVANCE(NonGeoDataset):
    """ADVANCE dataset.

    The `ADVANCE <https://akchen.github.io/ADVANCE-DATASET/>`__ dataset is a
    dataset for audio visual scene recognition.

    Dataset features:

    * 5,075 pairs of geotagged audio recordings and images
    * three spectral bands - RGB (512x512 px)
    * 10-second audio recordings

    Dataset classes:

    0. airport, 1. beach, 2. bridge, 3. farmland, 4. forest, 5. grassland,
    6. harbour, 7. lake, 8. orchard, 9. residential, 10. sparse shrub land,
    11. sports land, 12. train station

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1007/978-3-030-58586-0_5

    .. note::
        This dataset requires ``scipy`` to be installed to load the audio
        files.
    """

    urls = (
        "https://zenodo.org/records/3828124/files/ADVANCE_vision.zip?download=1",
        "https://zenodo.org/records/3828124/files/ADVANCE_sound.zip?download=1",
    )
    filenames = ("ADVANCE_vision.zip", "ADVANCE_sound.zip")
    md5s = ("a9e8748219ef5864d3b5a8979a67b471", "a2d12f2d2a64f5c3d3a9d8c09aaf1c31")
    directories = ("vision", "sound")
    classes = (
        "airport",
        "beach",
        "bridge",
        "farmland",
        "forest",
        "grassland",
        "harbour",
        "lake",
        "orchard",
        "residential",
        "sparse shrub land",
        "sports land",
        "train station",
    )

    def __init__(self, root="data", transforms=None, download=False, checksum=True):
        """Initialize a new ADVANCE dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
            DependencyNotFoundError: If scipy is not installed.
        """
        lazy_import("scipy.io.wavfile")

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        self.files = self._load_files(self.root)
        self.classes = tuple(sorted({f["cls"] for f in self.files}))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, index):
        files = self.files[index]
        image = self._load_image(files["image"])
        audio = self._load_target(files["audio"])
        cls_label = self.class_to_idx[files["cls"]]
        label = ops.convert_to_tensor(np.array(cls_label, dtype="int64"))
        sample = {"image": image, "audio": audio, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _load_files(self, root):
        images = sorted(glob.glob(os.path.join(root, "vision", "**", "*.jpg")))
        wavs = sorted(glob.glob(os.path.join(root, "sound", "**", "*.wav")))
        labels = [image.split(os.sep)[-2] for image in images]
        files = [
            {"image": image, "audio": wav, "cls": label}
            for image, wav, label in zip(images, wavs, labels)
        ]
        return files

    def _load_image(self, path):
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
            tensor = ops.cast(ops.convert_to_tensor(array), "float32")
            return tensor

    def _load_target(self, path):
        siw = lazy_import("scipy.io.wavfile")
        array = siw.read(path, mmap=True)[1]
        tensor = ops.convert_to_tensor(np.array(array))
        tensor = ops.expand_dims(tensor, axis=0)
        return tensor

    def _check_integrity(self):
        for directory in self.directories:
            filepath = os.path.join(self.root, directory)
            if not os.path.exists(filepath):
                return False
        return True

    def _download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for filename, url, md5 in zip(self.filenames, self.urls, self.md5s):
            download_and_extract_archive(
                url, self.root, filename=filename, md5=md5 if self.checksum else None
            )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"])
        label = int(ops.convert_to_numpy(sample["label"]))
        label_class = self.classes[label]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = int(ops.convert_to_numpy(sample["prediction"]))
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image.astype("uint8"))
        ax.axis("off")
        if show_titles:
            title = f"Label: {label_class}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_class}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
