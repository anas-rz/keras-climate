"""PASTIS dataset (ported from torchgeo.datasets.pastis)."""

import os

import geopandas as gpd
import numpy as np
from keras import ops
from matplotlib.colors import ListedColormap

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import check_integrity, download_url, extract_archive, quantile_normalization


class PASTIS(NonGeoDataset):
    """PASTIS dataset.

    The `PASTIS <https://github.com/VSainteuf/pastis-benchmark>`__ dataset
    is a dataset for time-series panoptic segmentation of agricultural
    parcels.

    Dataset features:

    * support for the original PASTIS and PASTIS-R versions of the dataset
    * 2,433 time-series with 10 m per pixel resolution (128x128 px)
    * 18 crop categories, 1 background category, 1 void category
    * semantic and instance annotations
    * 3 Sentinel-1 Ascending bands
    * 3 Sentinel-1 Descending bands
    * 10 Sentinel-2 L2A multispectral bands

    Dataset format:

    * time-series and annotations are in numpy format (.npy)

    Dataset classes:

    0. Background
    1. Meadow
    2. Soft Winter Wheat
    3. Corn
    4. Winter Barley
    5. Winter Rapeseed
    6. Spring Barley
    7. Sunflower
    8. Grapevine
    9. Beet
    10. Winter Triticale
    11. Winter Durum Wheat
    12. Fruits Vegetables Flowers
    13. Potatoes
    14. Leguminous Fodder
    15. Soybeans
    16. Orchard
    17. Mixed Cereal
    18. Sorghum
    19. Void Label

    If you use this dataset in your research, please cite the following
    papers:

    * https://doi.org/10.1109/ICCV48922.2021.00483
    * https://doi.org/10.1016/j.isprsjprs.2022.03.012
    """

    classes = (
        "background",  # all non-agricultural land
        "meadow",
        "soft_winter_wheat",
        "corn",
        "winter_barley",
        "winter_rapeseed",
        "spring_barley",
        "sunflower",
        "grapevine",
        "beet",
        "winter_triticale",
        "winter_durum_wheat",
        "fruits_vegetables_flowers",
        "potatoes",
        "leguminous_fodder",
        "soybeans",
        "orchard",
        "mixed_cereal",
        "sorghum",
        "void_label",  # for parcels mostly outside their patch
    )
    cmap = ListedColormap(
        np.array(
            [
                (0, 0, 0, 255),
                (174, 199, 232, 255),
                (255, 127, 14, 255),
                (255, 187, 120, 255),
                (44, 160, 44, 255),
                (152, 223, 138, 255),
                (214, 39, 40, 255),
                (255, 152, 150, 255),
                (148, 103, 189, 255),
                (197, 176, 213, 255),
                (140, 86, 75, 255),
                (196, 156, 148, 255),
                (227, 119, 194, 255),
                (247, 182, 210, 255),
                (127, 127, 127, 255),
                (199, 199, 199, 255),
                (188, 189, 34, 255),
                (219, 219, 141, 255),
                (23, 190, 207, 255),
                (255, 255, 255, 255),
            ]
        )
        / 255
    )
    directory = "PASTIS-R"
    filename = "PASTIS-R.zip"
    url = "https://zenodo.org/records/5735646/files/PASTIS-R.zip?download=1"
    md5 = "4887513d6c2d2b07fa935d325bd53e09"
    prefix = {
        "s2": os.path.join("DATA_S2", "S2_"),
        "s1a": os.path.join("DATA_S1A", "S1A_"),
        "s1d": os.path.join("DATA_S1D", "S1D_"),
        "semantic": os.path.join("ANNOTATIONS", "TARGET_"),
        "instance": os.path.join("INSTANCE_ANNOTATIONS", "INSTANCES_"),
    }
    s2_bands = (
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    )
    s1a_bands = ("S1A_VV", "S1A_VH", "S1A_VV_VH")
    s1d_bands = ("S1D_VV", "S1D_VH", "S1D_VV_VH")

    def __init__(
        self,
        root="data",
        folds=(1, 2, 3, 4, 5),
        bands=s2_bands,
        mode="semantic",
        transforms=None,
        download=False,
        checksum=True,
    ):
        """Initialize a new PASTIS dataset instance.

        Args:
            root: root directory where dataset can be found
            folds: a sequence of integers from 1 to 5 specifying which of
                the five dataset folds to include
            bands: sequence of band names to load. Must be a non-empty
                subset of ``s2_bands``, ``s1a_bands``, or ``s1d_bands``. All
                bands must come from the same sensor. Defaults to all S2
                bands.
            mode: load semantic ("semantic") or instance ("instance")
                annotations
            transforms: a function/transform that takes input sample and its
                target as entry and returns a transformed version
            download: if True, download dataset and store it in the root
                directory
            checksum: if True, check the MD5 of the downloaded files (may be
                slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is
                False.
        """
        for fold in folds:
            assert 1 <= fold <= 5
        assert mode in ["semantic", "instance"]

        bands_set = set(bands)
        if not bands_set:
            raise ValueError("bands must not be empty")
        if bands_set <= set(self.s2_bands):
            self.image_key = "s2"
            all_bands = self.s2_bands
        elif bands_set <= set(self.s1a_bands):
            self.image_key = "s1a"
            all_bands = self.s1a_bands
        elif bands_set <= set(self.s1d_bands):
            self.image_key = "s1d"
            all_bands = self.s1d_bands
        else:
            raise ValueError(
                f"bands must be a subset of s2_bands, s1a_bands, or s1d_bands; got {bands}"
            )
        self.bands = tuple(bands)
        self.band_indices = [all_bands.index(b) for b in self.bands]

        self.root = root
        self.folds = folds
        self.mode = mode
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self._verify()
        self.files = self._load_files()

    def __getitem__(self, index):
        """Return an index within the dataset."""
        image = self._load_image(index)
        if self.mode == "semantic":
            mask = self._load_semantic_targets(index)
            sample = {"image": image, "mask": mask}
        elif self.mode == "instance":
            mask, boxes, labels = self._load_instance_targets(index)
            sample = {"image": image, "mask": mask, "bbox_xyxy": boxes, "label": labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return the number of data points in the dataset."""
        return len(self.idxs)

    def _load_image(self, index):
        """Load a single time-series."""
        path = self.files[index][self.image_key]
        array = np.load(path)[:, self.band_indices, :, :]
        # (T, C, H, W) -> (T, H, W, C)
        array = np.transpose(array, (0, 2, 3, 1)).astype("float32")
        return ops.convert_to_tensor(array)

    def _load_semantic_targets(self, index):
        """Load the target mask for a single image."""
        # See https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader.py#L201
        # even though the mask file is 3 bands, we just select the first band
        array = np.load(self.files[index]["semantic"])[0].astype("int64")
        return ops.convert_to_tensor(array)

    def _load_instance_targets(self, index):
        """Load the instance segmentation targets for a single sample."""
        mask_array = np.load(self.files[index]["semantic"])[0]
        instance_array = np.load(self.files[index]["instance"])

        instance_ids = np.unique(instance_array)
        # Exclude a mask for unknown/background
        instance_ids = instance_ids[instance_ids != 0]

        if len(instance_ids) == 0:
            masks = np.zeros((0, *instance_array.shape), dtype="uint8")
            boxes = np.zeros((0, 4), dtype="float32")
            labels = np.zeros((0,), dtype="int64")
            return (
                ops.convert_to_tensor(masks),
                ops.convert_to_tensor(boxes),
                ops.convert_to_tensor(labels),
            )

        masks = instance_array == instance_ids[:, None, None]

        labels_list = []
        boxes_list = []
        for mask in masks:
            label = np.unique(mask_array[mask])[0]
            labels_list.append(label)

            pos = np.where(mask)
            xmin, xmax = pos[1].min(), pos[1].max()
            ymin, ymax = pos[0].min(), pos[0].max()
            boxes_list.append([xmin, ymin, xmax, ymax])

        masks = masks.astype("uint8")
        boxes = np.array(boxes_list, dtype="float32")
        labels = np.array(labels_list, dtype="int64")

        return (
            ops.convert_to_tensor(masks),
            ops.convert_to_tensor(boxes),
            ops.convert_to_tensor(labels),
        )

    def _load_files(self):
        """List the image and target files."""
        metadata_fn = os.path.join(self.root, self.directory, "metadata.geojson")
        gdf = gpd.read_file(metadata_fn)
        gdf["Fold"] = gdf["Fold"].astype(int)
        gdf = gdf[gdf["Fold"].isin(self.folds)]
        self.idxs = gdf["ID_PATCH"].tolist()

        files = []
        for i in self.idxs:
            path = os.path.join(self.root, self.directory, "{}") + str(i) + ".npy"
            files.append(
                {
                    "s2": path.format(self.prefix["s2"]),
                    "s1a": path.format(self.prefix["s1a"]),
                    "s1d": path.format(self.prefix["s1d"]),
                    "semantic": path.format(self.prefix["semantic"]),
                    "instance": path.format(self.prefix["instance"]),
                }
            )
        return files

    def _verify(self):
        """Verify the integrity of the dataset."""
        # Check if the directory already exists
        path = os.path.join(self.root, self.directory)
        if os.path.exists(path):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()

    def _download(self):
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
        extract_archive(os.path.join(self.root, self.filename), self.root)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        # Keep the RGB bands and quantile-normalize each frame independently.
        rgb_indices = np.array([2, 1, 0])
        rgb_frames = ops.take(sample["image"], rgb_indices, axis=-1)
        images = np.stack(
            [
                ops.convert_to_numpy(quantile_normalization(frame))
                for frame in rgb_frames
            ]
        )
        mask = ops.convert_to_numpy(sample["mask"])

        if self.mode == "instance":
            label = ops.convert_to_numpy(sample["label"])
            mask = label[mask.argmax(axis=0)]

        num_panels = 3
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = ops.convert_to_numpy(sample["prediction"])
            num_panels += 1
            if self.mode == "instance":
                predictions = predictions.argmax(axis=0)
                pred_label = ops.convert_to_numpy(sample["prediction_labels"])
                predictions = pred_label[predictions]

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 4))
        axs[0].imshow(images[0])
        axs[1].imshow(images[1])
        axs[2].imshow(mask, vmin=0, vmax=19, cmap=self.cmap, interpolation="none")
        axs[0].axis("off")
        axs[1].axis("off")
        axs[2].axis("off")
        if showing_predictions:
            axs[3].imshow(predictions, vmin=0, vmax=19, cmap=self.cmap, interpolation="none")
            axs[3].axis("off")

        if show_titles:
            axs[0].set_title("Image 0")
            axs[1].set_title("Image 1")
            axs[2].set_title("Mask")
            if showing_predictions:
                axs[3].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class PASTIS100(PASTIS):
    """Subset of PASTIS-R containing only 100 time-series.

    Intended for tutorials and demonstrations, not for benchmarking.
    """

    directory = "PASTIS-R"
    filename = "PASTIS-R.zip"
    url = "https://huggingface.co/datasets/torchgeo/PASTIS-R-100/resolve/acd0180e834e40934b79c0121f606d4f8ca3299d/PASTIS-R.zip"
    md5 = "6b4a428bd27cdbc2abda44973ba42892"
