"""SkyScript dataset (ported from torchgeo.datasets.skyscript)."""

import pathlib
import textwrap

import numpy as np
import pandas as pd
from keras import ops
from PIL import Image

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive, download_url, extract_archive, lazy_import


class SkyScript(NonGeoDataset):
    """SkyScript dataset.

    `SkyScript <https://github.com/wangzhecheng/SkyScript>`__ is a large and
    semantically diverse image-text dataset for remote sensing images. It contains
    5.2 million remote sensing image-text pairs in total, covering more than 29K
    distinct semantic tags.

    If you use this dataset in your research, please cite it using the following format:

    * https://arxiv.org/abs/2312.12856

    .. note::
       This dataset requires the following additional library to be installed:

       * `tokenizers <https://pypi.org/project/tokenizers/>`_ to tokenize the captions
    """

    url = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/{}"

    image_dirs = tuple(f"images{i}" for i in range(2, 8))
    image_md5s = (
        "fbfb5f7aa1731f4106fc3ffbd608100a",
        "ad4fd9fdb9622d1ea360210cb222f2bd",
        "aeeb41e830304c74b14b5ffc1fc8e8c3",
        "02ee7e0e59f9ac1c87b678a155e1f1df",
        "350475f1e7fb996152fa16db891b4142",
        "5e2fbf3e9262b36e30b458ec9a1df625",
    )

    #: Can be modified in subclasses to change train/val/test split
    caption_files = {
        "train": "SkyScript_train_top30pct_filtered_by_CLIP_openai.csv",
        "val": "SkyScript_val_5K_filtered_by_CLIP_openai.csv",
        "test": "SkyScript_test_30K_filtered_by_CLIP_openai.csv",
    }
    caption_md5s = {
        "train": "05b362e43a852667b5374c9a5ae53f8e",
        "val": "c8d278fd29b754361989d5e7a6608f69",
        "test": "0135d9b49ce6751360912a4353e809dc",
    }

    def __init__(
        self,
        root="data",
        split="train",
        transforms=None,
        download=False,
        checksum=True,
        tokenizer=None,
    ):
        """Initialize a new SkyScript instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', 'test'.
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).
            tokenizer: A pre-trained tokenizer (defaults to a BPE tokenizer trained
                on the training split's captions).

        Raises:
            AssertionError: If *split* is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
            DependencyNotFoundError: If tokenizers is not installed.
        """
        assert split in self.caption_files

        self.root = pathlib.Path(root)
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.captions = pd.read_csv(self.root / self.caption_files[split])

        if tokenizer is None:
            tokenizers = lazy_import("tokenizers")
            self.tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
            trainer = tokenizers.trainers.BpeTrainer()
            train_captions = pd.read_csv(self.root / self.caption_files["train"])
            self.tokenizer.train_from_iterator(train_captions["title"], trainer)
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.captions)

    def __getitem__(self, index):
        """Return an index within the dataset.

        Returns:
            A dict containing image and tokenized caption at index.
        """
        filepath, title = self.captions.iloc[index][:2]
        output = self.tokenizer.encode(title)

        with Image.open(self.root / filepath) as img:
            array = np.array(img, dtype=np.float32)

        sample = {
            "image": ops.convert_to_tensor(array),
            "caption": ops.convert_to_tensor(np.array(output.ids, dtype="int64")),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self):
        """Verify the integrity of the dataset."""
        for directory, md5 in zip(self.image_dirs, self.image_md5s):
            # Check if the extracted files already exist
            if (self.root / directory).is_dir():
                continue

            # Check if the zip files have already been downloaded
            if (self.root / f"{directory}.zip").is_file():
                extract_archive(self.root / f"{directory}.zip")
                continue

            # Check if the user requested to download the dataset
            if not self.download:
                raise DatasetNotFoundError(self)

            # Download the dataset
            url = self.url.format(f"{directory}.zip")
            md5 = md5 if self.checksum else None
            download_and_extract_archive(url, self.root, md5=md5)

        # Download the caption files
        for split in {"train", self.split}:
            if self.download:
                url = self.url.format(self.caption_files[split])
                md5 = self.caption_md5s[split] if self.checksum else None
                download_url(url, self.root, md5=md5)

    def plot(self, sample, show_titles=True, suptitle=None):
        """Plot a sample from the dataset."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        image = ops.convert_to_numpy(sample["image"]) / 255
        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            caption = ops.convert_to_numpy(sample["caption"]).tolist()
            title = textwrap.wrap(self.tokenizer.decode(caption))
            if "prediction" in sample:
                caption = ops.convert_to_numpy(sample["prediction"]).tolist()
                title += textwrap.wrap(self.tokenizer.decode(caption))
            ax.set_title("\n".join(title))

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
