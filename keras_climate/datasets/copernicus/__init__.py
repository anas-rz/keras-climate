"""Copernicus-Bench datasets (ported from torchgeo.datasets.copernicus)."""

from .aq_no2_s5p import CopernicusBenchAQNO2S5P
from .aq_o3_s5p import CopernicusBenchAQO3S5P
from .base import CopernicusBenchBase
from .bigearthnet_s1 import CopernicusBenchBigEarthNetS1
from .bigearthnet_s2 import CopernicusBenchBigEarthNetS2
from .biomass_s3 import CopernicusBenchBiomassS3
from .cloud_s2 import CopernicusBenchCloudS2
from .cloud_s3 import CopernicusBenchCloudS3
from .dfc2020_s1 import CopernicusBenchDFC2020S1
from .dfc2020_s2 import CopernicusBenchDFC2020S2
from .embed import CopernicusEmbed
from .eurosat_s1 import CopernicusBenchEuroSATS1
from .eurosat_s2 import CopernicusBenchEuroSATS2
from .flood_s1 import CopernicusBenchFloodS1
from .lc100cls_s3 import CopernicusBenchLC100ClsS3
from .lc100seg_s3 import CopernicusBenchLC100SegS3
from .lcz_s2 import CopernicusBenchLCZS2
from .pretrain import CopernicusPretrain

__all__ = (
    "CopernicusBenchAQNO2S5P",
    "CopernicusBenchAQO3S5P",
    "CopernicusBenchBase",
    "CopernicusBenchBigEarthNetS1",
    "CopernicusBenchBigEarthNetS2",
    "CopernicusBenchBiomassS3",
    "CopernicusBenchCloudS2",
    "CopernicusBenchCloudS3",
    "CopernicusBenchDFC2020S1",
    "CopernicusBenchDFC2020S2",
    "CopernicusBenchEuroSATS1",
    "CopernicusBenchEuroSATS2",
    "CopernicusBenchFloodS1",
    "CopernicusBenchLC100ClsS3",
    "CopernicusBenchLC100SegS3",
    "CopernicusBenchLCZS2",
    "CopernicusEmbed",
    "CopernicusPretrain",
)
