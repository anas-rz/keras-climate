"""GID-15 dataset (ported from torchgeo.datasets.gid15)."""

import glob
import os

import numpy as np
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive


class GID15(NonGeoDataset):
    """GID-15 dataset.

    The `GID-15 <https://captain-whu.github.io/GID15/>`__ dataset is a
    dataset for semantic segmentation.

    Dataset features:

    * images taken by the Gaofen-2 (GF-2) satellite over 60 cities in China
    * masks representing 15 semantic categories
    * three spectral bands - RGB
    * 150 with 3 m per pixel resolution (6800x7200 px)

    Dataset format:

    * images are three-channel pngs
    * masks are single-channel pngs
    * colormapped masks are 3 channel tifs

    Dataset classes:

    1. background
    2. industrial_land
    3. urban_residential
    4. rural_residential
    5. traffic_land
    6. paddy_field
    7. irrigated_land
    8. dry_cropland
    9. garden_plot
    10. arbor_woodland
    11. shrub_land
    12. natural_grassland
    13. artificial_grassland
    14. river
    15. lake
    16. pond

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1016/j.rse.2019.111322
    """

    url = "https://drive.google.com/file/d/1zbkCEXPEKEV6gq19OKmIbaT8bXXfWW6u"
    md5 = "615682bf659c3ed981826c6122c10c83"
    filename = "gid-15.zip"
    directory = "GID"
    splits = ("train", "val", "test")
    classes = (
        "background",
        "industrial_land",
        "urban_residential",
        "rural_residential",
        "traffic_land",
        "paddy_field",
        "irrigated_land",
        "dry_cropland",
        "garden_plot",
        "arbor_woodland",
        "shrub_land",
        "natural_grassland",
        "artificial_grassland",
        "river",
        "lake",
        "pond",
    )

    def __init__(
        self,
        root="data",
        split="train",
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new GID-15 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert split in self.splits

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        self.files = self._load_files(self.root, self.split)

    def __getitem__(self, index):
        """Return an index within the dataset."""
        files = self.files[index]
        image = self._load_image(files["image"])

        if self.split != "test":
            mask = self._load_target(files["mask"])
            sample = {"image": image, "mask": mask}
        else:
            sample = {"image": image}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def _load_files(self, root, split):
        """Return the paths of the files in the dataset."""
        image_root = os.path.join(root, "GID", "img_dir")
        images = glob.glob(os.path.join(image_root, split, "*.tif"))
        images = sorted(images)
        if split != "test":
            masks = [
                image.replace("img_dir", "ann_dir").replace(".tif", "_15label.png")
                for image in images
            ]
            files = [{"image": image, "mask": mask} for image, mask in zip(images, masks)]
        else:
            files = [{"image": image} for image in images]

        return files

    def _load_image(self, path):
        """Load a single image (returned channels-last as HxWxC)."""
        from PIL import Image

        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("RGB"))
            return ops.convert_to_tensor(array.astype("float32"))

    def _load_target(self, path):
        """Load the target mask for a single image."""
        from PIL import Image

        filename = os.path.join(path)
        with Image.open(filename) as img:
            array = np.array(img.convert("L"))
            return ops.convert_to_tensor(array.astype("int64"))

    def _check_integrity(self):
        """Checks the integrity of the dataset structure."""
        filepath = os.path.join(self.root, self.directory)
        return os.path.exists(filepath)

    def _download(self):
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        if self.split != "test":
            image, mask = sample["image"], sample["mask"]
            ncols = 2
        else:
            image = sample["image"]
            ncols = 1

        if "prediction" in sample:
            ncols += 1
            pred = sample["prediction"]

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 10, 10))

        if self.split != "test":
            axs[0].imshow(ops.convert_to_numpy(image).astype("uint8"))
            axs[0].axis("off")
            axs[1].imshow(ops.convert_to_numpy(mask))
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Image")
                axs[1].set_title("Mask")

            if "prediction" in sample:
                axs[2].imshow(ops.convert_to_numpy(pred))
                axs[2].axis("off")
                if show_titles:
                    axs[2].set_title("Prediction")
        else:
            if "prediction" in sample:
                axs[0].imshow(ops.convert_to_numpy(image).astype("uint8"))
                axs[0].axis("off")
                axs[1].imshow(ops.convert_to_numpy(pred))
                axs[1].axis("off")
                if show_titles:
                    axs[0].set_title("Image")
                    axs[1].set_title("Prediction")
            else:
                axs.imshow(ops.convert_to_numpy(image).astype("uint8"))
                axs.axis("off")
                if show_titles:
                    axs.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
