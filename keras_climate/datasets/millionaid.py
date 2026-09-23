"""Million-AID dataset (ported from torchgeo.datasets.millionaid)."""

import glob
import os

import numpy as np
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive


class MillionAID(NonGeoDataset):
    """Million-AID Dataset.

    The `MillionAID <https://captain-whu.github.io/DiRS/>`_ dataset consists
    of one million aerial images from Google Earth Engine that offers either
    a multi-class learning task with 51 classes or a multi-label learning
    task with 73 different possible labels. For more details please consult
    the accompanying `paper <https://ieeexplore.ieee.org/document/9393553>`_.

    Dataset format:

    * images are three-channel jpg

    If you use this dataset in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/9393553
    """

    multi_label_categories = (
        "agriculture_land",
        "airport_area",
        "apartment",
        "apron",
        "arable_land",
        "bare_land",
        "baseball_field",
        "basketball_court",
        "beach",
        "bridge",
        "cemetery",
        "church",
        "commercial_area",
        "commercial_land",
        "dam",
        "desert",
        "detached_house",
        "dry_field",
        "factory_area",
        "forest",
        "golf_course",
        "grassland",
        "greenhouse",
        "ground_track_field",
        "helipad",
        "highway_area",
        "ice_land",
        "industrial_land",
        "intersection",
        "island",
        "lake",
        "leisure_land",
        "meadow",
        "mine",
        "mining_area",
        "mobile_home_park",
        "oil_field",
        "orchard",
        "paddy_field",
        "parking_lot",
        "pier",
        "port_area",
        "power_station",
        "public_service_land",
        "quarry",
        "railway",
        "railway_area",
        "religious_land",
        "residential_land",
        "river",
        "road",
        "rock_land",
        "roundabout",
        "runway",
        "solar_power_plant",
        "sparse_shrub_land",
        "special_land",
        "sports_land",
        "stadium",
        "storage_tank",
        "substation",
        "swimming_pool",
        "tennis_court",
        "terraced_field",
        "train_station",
        "transportation_land",
        "unutilized_land",
        "viaduct",
        "wastewater_plant",
        "water_area",
        "wind_turbine",
        "woodland",
        "works",
    )

    multi_class_categories = (
        "apartment",
        "apron",
        "bare_land",
        "baseball_field",
        "bapsketball_court",
        "beach",
        "bridge",
        "cemetery",
        "church",
        "commercial_area",
        "dam",
        "desert",
        "detached_house",
        "dry_field",
        "forest",
        "golf_course",
        "greenhouse",
        "ground_track_field",
        "helipad",
        "ice_land",
        "intersection",
        "island",
        "lake",
        "meadow",
        "mine",
        "mobile_home_park",
        "oil_field",
        "orchard",
        "paddy_field",
        "parking_lot",
        "pier",
        "quarry",
        "railway",
        "river",
        "road",
        "rock_land",
        "roundabout",
        "runway",
        "solar_power_plant",
        "sparse_shrub_land",
        "stadium",
        "storage_tank",
        "substation",
        "swimming_pool",
        "tennis_court",
        "terraced_field",
        "train_station",
        "viaduct",
        "wastewater_plant",
        "wind_turbine",
        "works",
    )

    md5s = {
        "train": "1b40503cafa9b0601653ca36cd788852",
        "test": "51a63ee3eeb1351889eacff349a983d8",
    }

    filenames = {"train": "train.zip", "test": "test.zip"}

    tasks = ("multi-class", "multi-label")
    splits = ("train", "test")

    def __init__(
        self,
        root="data",
        task="multi-class",
        split="train",
        transforms=None,
        checksum=True,
    ):
        """Initialize a new MillionAID dataset instance.

        Args:
            root: root directory where dataset can be found
            task: type of task, either "multi-class" or "multi-label"
            split: train or test split
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            checksum: if True, check the checksum of the downloaded files
                (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        assert task in self.tasks
        assert split in self.splits
        self.task = task
        self.split = split

        self._verify()

        self.files = self._load_files(self.root)

        self.classes = sorted({cls for f in self.files for cls in f["label"]})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.files)

    def __getitem__(self, index):
        """Return an index within the dataset."""
        files = self.files[index]
        image = self._load_image(files["image"])
        cls_label = [self.class_to_idx[label] for label in files["label"]]
        label = ops.convert_to_tensor(np.array(cls_label, dtype="int64"))
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_files(self, root):
        """Return the paths of the files in the dataset."""
        imgs_no_subcat = list(glob.glob(os.path.join(root, self.split, "*", "*", "*.jpg")))

        imgs_subcat = list(
            glob.glob(os.path.join(root, self.split, "*", "*", "*", "*.jpg"))
        )

        scenes = [p.split(os.sep)[-3] for p in imgs_no_subcat] + [
            p.split(os.sep)[-4] for p in imgs_subcat
        ]

        subcategories = ["Missing" for p in imgs_no_subcat] + [
            p.split(os.sep)[-3] for p in imgs_subcat
        ]

        classes = [p.split(os.sep)[-2] for p in imgs_no_subcat] + [
            p.split(os.sep)[-2] for p in imgs_subcat
        ]

        if self.task == "multi-label":
            labels = [
                [sc, sub, c] if sub != "Missing" else [sc, c]
                for sc, sub, c in zip(scenes, subcategories, classes)
            ]
        else:
            labels = [[c] for c in classes]

        images = imgs_no_subcat + imgs_subcat

        files = [{"image": image, "label": label} for image, label in zip(images, labels)]

        return files

    def _load_image(self, path):
        """Load a single image, returned channels-last (H, W, C)."""
        with Image.open(path) as img:
            array = np.array(img.convert("RGB"))
            return ops.convert_to_tensor(array)

    def _verify(self):
        """Verify the integrity of the dataset."""
        filepath = os.path.join(self.root, self.split)
        if os.path.isdir(filepath):
            return

        filepath = os.path.join(self.root, self.split + ".zip")
        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5s[self.split]):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        raise DatasetNotFoundError(self)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        image = ops.convert_to_numpy(sample["image"]).astype("uint8")
        labels = [self.classes[int(label)] for label in ops.convert_to_numpy(sample["label"])]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_labels = [
                self.classes[int(label)] for label in ops.convert_to_numpy(sample["prediction"])
            ]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {labels}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_labels}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
