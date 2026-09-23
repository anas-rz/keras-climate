"""TreeSatAI datasets (ported from torchgeo.datasets.treesatai)."""

import json
import os

import numpy as np
import rasterio as rio
from keras import ops

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, extract_archive, quantile_normalization


class TreeSatAI(NonGeoDataset):
    """TreeSatAI Benchmark Archive.

    `TreeSatAI Benchmark Archive <https://zenodo.org/records/6780578>`_ is
    a multi-sensor, multi-label dataset for tree species classification in
    remote sensing. It was created by combining labels from the federal
    forest inventory of Lower Saxony, Germany with 20 cm Color-Infrared
    (CIR) and 10 m Sentinel imagery.

    The TreeSatAI Benchmark Archive contains:

    * 50,381 image triplets (aerial, Sentinel-1, Sentinel-2)
    * synchronized time steps and locations
    * all original spectral bands/polarizations from the sensors
    * 20 species classes (single labels)
    * 12 age classes (single labels)
    * 15 genus classes (multi labels)
    * 60 m and 200 m patches
    * fixed split for train (90%) and test (10%) data
    * additional single labels such as English species name, genus,
      forest stand type, foliage type, land cover

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.5194/essd-15-681-2023
    """

    url = "https://zenodo.org/records/6780578/files/"
    md5s = {
        "aerial_60m_abies_alba.zip": "4298b1c9fbf6d0d85f7aa208ff5fe0c9",
        "aerial_60m_acer_pseudoplatanus.zip": "7c31d7ddea841f6509deece8f984a79e",
        "aerial_60m_alnus_spec.zip": "34ea107f43c6172c6d2652dbf26306af",
        "aerial_60m_betula_spec.zip": "69de9373739a027692a823846434fa0c",
        "aerial_60m_cleared.zip": "8dffbb2f6aad17ef83721cffa5b52d96",
        "aerial_60m_fagus_sylvatica.zip": "77b277e69e90bfbd3c5fd15a73d228fe",
        "aerial_60m_fraxinus_excelsior.zip": "9a88a8e6821f8a54ded950de9238831f",
        "aerial_60m_larix_decidua.zip": "aa0bc5b091b099018a078536ef429031",
        "aerial_60m_larix_kaempferi.zip": "429df073f69f8bbf60aef765e1c925ba",
        "aerial_60m_picea_abies.zip": "edb9b1bc9a5a7b405f4cbb0d71cedf54",
        "aerial_60m_pinus_nigra.zip": "96bf1798ef82f712ea46c2963ddb7083",
        "aerial_60m_pinus_strobus.zip": "0ff818c6d31f59b8488880e49b300c7a",
        "aerial_60m_pinus_sylvestris.zip": "298cbaac4d9f07a204e1e74e8446798d",
        "aerial_60m_populus_spec.zip": "46fcff76b119cc24f3caf938a0bb433a",
        "aerial_60m_prunus_spec.zip": "fb1c570d3ea925a049630224ccb354bc",
        "aerial_60m_pseudotsuga_menziesii.zip": "2d05511ceabf4037b869eca928f3c04e",
        "aerial_60m_quercus_petraea.zip": "31f573fb0419b2b453ed7da1c4d2a298",
        "aerial_60m_quercus_robur.zip": "bcd90506509de26692c043f4c8d73af0",
        "aerial_60m_quercus_rubra.zip": "71d8495725ed1b4f27d9e382409fcc5e",
        "aerial_60m_tilia_spec.zip": "f81558c9c7189ac8a257d041ee43c1c9",
        "geojson.zip": "aa749718f3cb76c1dfc9cddc2ed201db",
        "labels.zip": "656f1b68ec9ab70afd02bb127b75bb24",
        "s1.zip": "bed4fc8cb65da46a24ec1bc6cea2763c",
        "s2.zip": "453ba69056aa33a3c6b97afb7b6afadb",
        "test_filenames.lst": "2166903d947f0025f61e342da466f917",
        "train_filenames.lst": "a1a0148e8120b0268f76d2e98a68436f",
    }

    # Genus-level classes (species-level labels also exist)
    classes = (
        "Abies",  # fir
        "Acer",  # maple
        "Alnus",  # alder
        "Betula",  # birch
        "Cleared",  # none
        "Fagus",  # beech
        "Fraxinus",  # ash
        "Larix",  # larch
        "Picea",  # spruce
        "Pinus",  # pine
        "Populus",  # poplar
        "Prunus",  # cherry
        "Pseudotsuga",  # Douglas fir
        "Quercus",  # oak
        "Tilia",  # linden
    )

    # https://zenodo.org/records/6780578/files/220629_doc_TreeSatAI_benchmark_archive.pdf
    all_sensors = ("aerial", "s1", "s2")
    all_bands_dict = {
        "aerial": ("IR", "G", "B", "R"),
        "s1": ("VV", "VH", "VV/VH"),
        "s2": (
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ),
    }
    rgb_bands_dict = {
        "aerial": ("R", "G", "B"),
        "s1": ("VV", "VH", "VV/VH"),
        "s2": ("B04", "B03", "B02"),
    }

    def __init__(
        self,
        root="data",
        split="train",
        sensors=all_sensors,
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new TreeSatAI instance.

        Args:
            root: root directory where dataset can be found
            split: either "train" or "test"
            sensors: one or more of "aerial", "s1", and/or "s2"
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            AssertionError: If invalid *sensors* are chosen.
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        assert set(sensors) <= set(self.all_sensors)

        self.root = root
        self.split = split
        self.sensors = sensors
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        path = os.path.join(self.root, f"{split}_filenames.lst")
        with open(path) as f:
            self.files = f.read().strip().split("\n")

        path = os.path.join(self.root, "labels", "TreeSatBA_v9_60m_multi_labels.json")
        with open(path) as f:
            self.labels = json.load(f)

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def __getitem__(self, index):
        """Return an index within the dataset."""
        file = self.files[index]
        label = np.zeros(len(self.classes), dtype="int64")
        for genus, _ in self.labels[file]:
            i = self.classes.index(genus)
            label[i] = 1

        sample = {"label": ops.convert_to_tensor(label)}
        for directory in self.sensors:
            with rio.open(os.path.join(self.root, directory, "60m", file)) as f:
                array = f.read().astype("float32")
                # (C, H, W) -> (H, W, C)
                array = np.transpose(array, (1, 2, 0))
                sample[f"image_{directory}"] = ops.convert_to_tensor(array)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        exists = []
        for directory in self.sensors:
            exists.append(os.path.isdir(os.path.join(self.root, directory)))

        if all(exists):
            return

        for file, md5 in self.md5s.items():
            # Check if the file has already been downloaded
            if os.path.isfile(os.path.join(self.root, file)):
                self._extract(file)
                continue

            # Check if the user requested to download the dataset
            if self.download:
                url = self.url + file
                download_url(url, self.root, md5=md5 if self.checksum else None)
                self._extract(file)
                continue

            raise DatasetNotFoundError(self)

    def _extract(self, file):
        """Extract file."""
        if not file.endswith(".zip"):
            return

        to_path = self.root
        if file.startswith("aerial"):
            to_path = os.path.join(self.root, "aerial", "60m")

        extract_archive(os.path.join(self.root, file), to_path)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(ncols=len(self.sensors), squeeze=False)

        for i, sensor in enumerate(self.sensors):
            image = ops.convert_to_numpy(sample[f"image_{sensor}"])
            bands = [
                self.all_bands_dict[sensor].index(b)
                for b in self.rgb_bands_dict[sensor]
            ]
            image = np.take(image, bands, axis=-1)
            image = ops.convert_to_numpy(quantile_normalization(image))
            ax[0, i].imshow(image)
            ax[0, i].axis("off")

            if show_titles:
                ax[0, i].set_title(sensor)

        suptitle_str = ""

        if suptitle is not None:
            suptitle_str = suptitle

        if show_titles:
            label = self._multilabel_to_string(sample["label"])
            if suptitle_str:
                suptitle_str += f"\nLabel: ({label})"
            else:
                suptitle_str = f"Label: ({label})"

            if "prediction" in sample:
                prediction = self._multilabel_to_string(sample["prediction"])
                suptitle_str += f"\nPrediction: ({prediction})"

        if suptitle_str:
            plt.suptitle(suptitle_str)

        fig.tight_layout()
        return fig

    def _multilabel_to_string(self, multilabel):
        """Convert a tensor of multilabel class probabilities to human readable format."""
        multilabel = ops.convert_to_numpy(multilabel)
        labels = []
        for i, pct in enumerate(multilabel):
            if pct > 0.001:
                labels.append((self.classes[i], float(pct)))

        labels.sort(key=lambda label: label[1], reverse=True)
        return ", ".join([f"{genus}: {pct:.1%}" for genus, pct in labels])
