"""SpaceNet 8 dataset (ported from torchgeo.datasets.spacenet.spacenet8)."""

from .base import SpaceNet


class SpaceNet8(SpaceNet):
    """SpaceNet8: Flood Detection Challenge Using Multiclass Segmentation.

    `SpaceNet 8 <https://spacenet.ai/sn8-challenge/>`_ is a dataset focusing on
    infrastructure and flood mapping related to hurricanes and heavy rains that cause
    route obstructions and significant damage.

    If you use this dataset in your research, please cite the following paper:

    * https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Hansch_SpaceNet_8\\_-_The_Detection_of_Flooded_Roads_and_Buildings_CVPRW_2022_paper.html
    """

    directory_glob = "{product}"
    file_regex = r"(\d+_\d+_\d+)\."
    dataset_id = "SN8_floods"
    tarballs = {
        "train": {
            0: [
                "Germany_Training_Public.tar.gz",
                "Louisiana-East_Training_Public.tar.gz",
            ]
        },
        "test": {0: ["Louisiana-West_Test_Public.tar.gz"]},
    }
    md5s = {
        "train": {
            0: ["81383a9050b93e8f70c8557d4568e8a2", "fa40ae3cf6ac212c90073bf93d70bd95"]
        },
        "test": {0: ["d41d8cd98f00b204e9800998ecf8427e"]},
    }
    valid_aois = {"train": [0], "test": [0]}
    valid_images = {
        "train": ["PRE-event", "POST-event"],
        "test": ["PRE-event", "POST-event"],
    }
    valid_masks = ("annotations",)
    chip_size = {
        "PRE-event": (1300, 1300),
        "POST-event": (1300, 1300),
    }
