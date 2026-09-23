"""GeoNRW dataset (ported from torchgeo.datasets.geonrw)."""

import os
from glob import glob

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive, extract_archive


class GeoNRW(NonGeoDataset):
    """GeoNRW dataset.

    This dataset contains RGB, DEM and segmentation label data from North
    Rhine-Westphalia, Germany.

    Dataset features:

    * 7298 training and 485 test samples
    * RGB images, 1000x1000px normalized to [0, 1]
    * DEM images, unnormalized
    * segmentation labels

    Dataset format:

    * RGB images are three-channel jp2
    * DEM images are single-channel tif
    * segmentation labels are single-channel tif

    Dataset classes:

    0. background
    1. forest
    2. water
    3. agricultural
    4. residential,commercial,industrial
    5. grassland,swamp,shrubbery
    6. railway,trainstation
    7. highway,squares
    8. airport,shipyard
    9. roads
    10. buildings

    Additional information about the dataset can be found
    `on this site <https://ieee-dataport.org/open-access/geonrw>`__.

    If you use this dataset in your research, please cite:

    * https://ieeexplore.ieee.org/document/9406194
    """

    # Splits taken from https://github.com/gbaier/geonrw/blob/ecfcdbca8cfaaeb490a9c6916980f385b9f3941a/pytorch/nrw.py#L48

    splits = ("train", "test")

    train_list = (
        "aachen",
        "bergisch",
        "bielefeld",
        "bochum",
        "bonn",
        "borken",
        "bottrop",
        "coesfeld",
        "dortmund",
        "dueren",
        "duisburg",
        "ennepetal",
        "erftstadt",
        "essen",
        "euskirchen",
        "gelsenkirchen",
        "guetersloh",
        "hagen",
        "hamm",
        "heinsberg",
        "herford",
        "hoexter",
        "kleve",
        "koeln",
        "krefeld",
        "leverkusen",
        "lippetal",
        "lippstadt",
        "lotte",
        "moenchengladbach",
        "moers",
        "muelheim",
        "muenster",
        "oberhausen",
        "paderborn",
        "recklinghausen",
        "remscheid",
        "siegen",
        "solingen",
        "wuppertal",
    )

    test_list = ("duesseldorf", "herne", "neuss")

    classes = (
        "background",
        "forest",
        "water",
        "agricultural",
        "residential,commercial,industrial",
        "grassland,swamp,shrubbery",
        "railway,trainstation",
        "highway,squares",
        "airport,shipyard",
        "roads",
        "buildings",
    )

    colormap = mcolors.ListedColormap(
        [
            "#000000",  # matplotlib black for background
            "#2ca02c",  # matplotlib green for forest
            "#1f77b4",  # matplotlib blue for water
            "#8c564b",  # matplotlib brown for agricultural
            "#7f7f7f",  # matplotlib gray residential_commercial_industrial
            "#bcbd22",  # matplotlib olive for grassland_swamp_shrubbery
            "#ff7f0e",  # matplotlib orange for railway_trainstation
            "#9467bd",  # matplotlib purple for highway_squares
            "#17becf",  # matplotlib cyan for airport_shipyard
            "#d62728",  # matplotlib red for roads
            "#e377c2",  # matplotlib pink for buildings
        ]
    )

    readers = {
        "rgb": lambda path: Image.open(path).convert("RGB"),
        "dem": lambda path: Image.open(path).copy(),
        "seg": lambda path: Image.open(path).convert("I;16"),
    }

    modality_filenames = {
        "rgb": lambda utm_coords: "{}_{}_rgb.jp2".format(*utm_coords),
        "dem": lambda utm_coords: "{}_{}_dem.tif".format(*utm_coords),
        "seg": lambda utm_coords: "{}_{}_seg.tif".format(*utm_coords),
    }

    modalities = ("rgb", "dem", "seg")

    url = "https://hf.co/datasets/torchgeo/geonrw/resolve/3cb6bdf2a615b9e526c7dcff85fd1f20728081b7/{}"

    filename = "nrw_dataset.tar.gz"
    sha256 = "de3cdda6a5718a8ae7f0e1479829a06c0fcebdab4eaf70274c56adfde696b972"

    def __init__(self, root="data", split="train", transforms=None, download=False, checksum=True):
        """Initialize the GeoNRW dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train", or "test"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, verify the checksum of the downloaded files
                (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert split in self.splits, f"split must be one of {self.splits}"

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self.city_names = self.test_list if split == "test" else self.train_list

        self._verify()

        self.file_list = self._get_file_list()

    def _get_file_list(self):
        """Get a list of files for cities in the dataset split."""
        file_list = []
        for cn in self.city_names:
            pattern = os.path.join(self.root, cn, "*rgb.jp2")
            file_list.extend(glob(pattern))
        return sorted(file_list)

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.file_list)

    def __getitem__(self, index):
        """Return an index within the dataset.

        Returns an "image" (H, W, 3) normalized to [0, 1], a "dem" (H, W, 1)
        with unnormalized values, and a "mask" (H, W).
        """
        path = self.file_list[index]
        utm_coords = os.path.basename(path).split("_")[:2]
        base_dir = os.path.dirname(path)

        sample = {}
        for modality in self.modalities:
            modality_path = os.path.join(base_dir, self.modality_filenames[modality](utm_coords))
            img = self.readers[modality](modality_path)
            array = np.array(img)

            if modality == "rgb":
                array = array.astype("float32") / 255.0
            elif modality == "dem":
                array = array.astype("float32")
                if array.ndim == 2:
                    array = array[..., None]
            elif modality == "seg":
                array = array.astype("int64")

            sample[modality] = ops.convert_to_tensor(array)

        # rename to keras_climate standard keys
        sample["image"] = sample.pop("rgb")
        sample["mask"] = sample.pop("seg")

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        # check if city names directories exist
        all_exist = all(os.path.exists(os.path.join(self.root, cn)) for cn in self.city_names)
        if all_exist:
            return

        # Check if the tar file has been downloaded
        if os.path.exists(os.path.join(self.root, self.filename)):
            extract_archive(os.path.join(self.root, self.filename), self.root)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self):
        """Download the dataset."""
        download_and_extract_archive(
            self.url.format(self.filename),
            download_root=self.root,
            sha256=self.sha256 if self.checksum else None,
        )

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        showing_predictions = "prediction" in sample
        ncols = 3
        if showing_predictions:
            prediction = ops.convert_to_numpy(sample["prediction"]).astype("int64")
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 10), sharex=True)

        axs[0].imshow(ops.convert_to_numpy(sample["image"]))
        axs[0].axis("off")
        axs[1].imshow(ops.convert_to_numpy(sample["dem"])[..., 0], cmap="gray")
        axs[1].axis("off")
        axs[2].imshow(
            ops.convert_to_numpy(sample["mask"]),
            cmap=self.colormap,
            vmin=0,
            vmax=10,
            interpolation="none",
        )
        axs[2].axis("off")

        if showing_predictions:
            axs[3].imshow(prediction, cmap=self.colormap, vmin=0, vmax=10, interpolation="none")

        # show classes in legend
        if show_titles:
            patches = [matplotlib.patches.Patch(color=c) for c in self.colormap.colors]
            axs[2].legend(patches, self.classes, loc="center left", bbox_to_anchor=(1, 0.5))

        if show_titles:
            axs[0].set_title("RGB Image")
            axs[1].set_title("DEM")
            axs[2].set_title("Labels")

        if suptitle is not None:
            fig.suptitle(suptitle, y=0.8)

        fig.tight_layout()

        return fig
