"""SpaceNet 6 dataset (ported from torchgeo.datasets.spacenet.spacenet6)."""

from .base import SpaceNet


class SpaceNet6(SpaceNet):
    """SpaceNet 6: Multi-Sensor All-Weather Mapping.

    `SpaceNet 6 <https://spacenet.ai/sn6-challenge/>`_ is a dataset
    of optical and SAR imagery over the city of Rotterdam.

    Collection features: Rotterdam covers 120 sq km, 3401 images, ~48000
    building footprint labels.

    Imagery features (GSD in meters, chip size in px):

    * PAN: 0.5 GSD, 900 x 900
    * RGBNIR: 2.0 GSD, 450 x 450
    * PS-RGB: 0.5 GSD, 900 x 900
    * PS-RGBNIR: 0.5 GSD, 900 x 900
    * SAR-Intensity: 0.5 GSD, 900 x 900

    Dataset format:

    * Imagery - GeoTIFFs from Worldview-2 (optical) and Capella Space (SAR)

        * PAN.tif (Panchromatic)
        * RGBNIR.tif (Multispectral)
        * PS-RGB (Pansharpened RGB)
        * PS-RGBNIR (Pansharpened RGBNIR)
        * SAR-Intensity (SAR Intensity)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2004.06500
    """

    file_regex = r"_tile_(\d+)\."
    dataset_id = "SN6_buildings"
    tarballs = {
        "train": {11: ["SN6_buildings_AOI_11_Rotterdam_train.tar.gz"]},
        "test": {11: ["SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz"]},
    }
    md5s = {
        "train": {11: ["10ca26d2287716e3b6ef0cf0ad9f946e"]},
        "test": {11: ["a07823a5e536feeb8bb6b6f0cb43cf05"]},
    }
    valid_aois = {"train": [11], "test": [11]}
    valid_images = {
        "train": ["PAN", "PS-RGB", "PS-RGBNIR", "RGBNIR", "SAR-Intensity"],
        "test": ["SAR-Intensity"],
    }
    valid_masks = ("geojson_buildings",)
