"""SpaceNet 5 dataset (ported from torchgeo.datasets.spacenet.spacenet5)."""

from .spacenet3 import SpaceNet3


class SpaceNet5(SpaceNet3):
    """SpaceNet 5: Automated Road Network Extraction and Route Travel Time Estimation.

    `SpaceNet 5 <https://spacenet.ai/sn5-challenge/>`_
    is a dataset of road networks over the cities of Moscow, Mumbai and San
    Juan (unavailable).

    Collection features (AOI, area km2, # images, road network labels in km):

    * Moscow: 1353, 1353, 3066
    * Mumbai: 1021, 1016, 1951

    Imagery features (GSD in meters, chip size in px):

    * PAN: 0.31 GSD, 1300 x 1300
    * MS: 1.24 GSD, 325 x 325
    * PS-MS: 0.30 GSD, 1300 x 1300
    * PS-RGB: 0.30 GSD, 1300 x 1300

    Dataset format:

    * Imagery - Worldview-3 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-MS (Pansharpened Multispectral)
        * PS-RGB (Pansharpened RGB)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please use the following citation:

    * The SpaceNet Partners, "SpaceNet5: Automated Road Network Extraction and
      Route Travel Time Estimation from Satellite Imagery",
      https://spacenet.ai/sn5-challenge/
    """

    file_regex = r"_chip(\d+)\."
    dataset_id = "SN5_roads"
    tarballs = {
        "train": {
            7: ["SN5_roads_train_AOI_7_Moscow.tar.gz"],
            8: ["SN5_roads_train_AOI_8_Mumbai.tar.gz"],
        },
        "test": {9: ["SN5_roads_test_public_AOI_9_San_Juan.tar.gz"]},
    }
    md5s = {
        "train": {
            7: ["03082d01081a6d8df2bc5a9645148d2a"],
            8: ["1ee20ba781da6cb7696eef9a95a5bdcc"],
        },
        "test": {9: ["fc45afef219dfd3a20f2d4fc597f6882"]},
    }
    valid_aois = {"train": [7, 8], "test": [9]}
    valid_images = {
        "train": ["MS", "PAN", "PS-MS", "PS-RGB"],
        "test": ["MS", "PAN", "PS-MS", "PS-RGB"],
    }
    valid_masks = ("geojson_roads_speed",)
