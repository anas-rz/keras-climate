"""Multi-backend Keras port of torchgeo.datasets.

Core framework: :class:`GeoDataset`, :class:`RasterDataset`,
:class:`VectorDataset`, :class:`XarrayDataset`, :class:`NonGeoDataset`,
:class:`NonGeoClassificationDataset`, :class:`IntersectionDataset`, and
:class:`UnionDataset`, plus dataset splitting utilities and collation
helpers, plus the full catalog of concrete leaf datasets ported from
torchgeo (EuroSAT, So2Sat, Sentinel, SpaceNet, Copernicus-Bench, ...).

Image/mask samples use the channels-last layout (``H x W x C``), a
deliberate deviation from torchgeo's channels-first convention to match the
rest of keras_climate. See :mod:`keras_climate.datasets.geo` for details.
"""

from .copernicus import (
    CopernicusBenchAQNO2S5P,
    CopernicusBenchAQO3S5P,
    CopernicusBenchBase,
    CopernicusBenchBigEarthNetS1,
    CopernicusBenchBigEarthNetS2,
    CopernicusBenchBiomassS3,
    CopernicusBenchCloudS2,
    CopernicusBenchCloudS3,
    CopernicusBenchDFC2020S1,
    CopernicusBenchDFC2020S2,
    CopernicusBenchEuroSATS1,
    CopernicusBenchEuroSATS2,
    CopernicusBenchFloodS1,
    CopernicusBenchLC100ClsS3,
    CopernicusBenchLC100SegS3,
    CopernicusBenchLCZS2,
    CopernicusEmbed,
    CopernicusPretrain,
)
from .errors import DatasetNotFoundError, DependencyNotFoundError, RGBBandsMissingError
from .eurosat import EuroSAT, EuroSAT100, EuroSATSpatial
from .geo import (
    GeoDataset,
    IntersectionDataset,
    NonGeoClassificationDataset,
    NonGeoDataset,
    RasterDataset,
    UnionDataset,
    VectorDataset,
    XarrayDataset,
)
from .mixins import PlottingMixin
from .spacenet import (
    SpaceNet,
    SpaceNet1,
    SpaceNet2,
    SpaceNet3,
    SpaceNet4,
    SpaceNet5,
    SpaceNet6,
    SpaceNet7,
    SpaceNet8,
)
from .splits import (
    random_bbox_assignment,
    random_bbox_splitting,
    random_grid_cell_assignment,
    roi_split,
    time_series_split,
)
from .utils import (
    BoundingBox,
    concat_samples,
    merge_samples,
    stack_samples,
    unbind_samples,
)

from .advance import ADVANCE
from .agb_live_woody_density import AbovegroundLiveWoodyBiomassDensity
from .agrifieldnet import AgriFieldNetImage, AgriFieldNetMask, AgriFieldNet
from .air_quality import AirQuality
from .airphen import Airphen
from .astergdem import AsterGDEM
from .benin_cashews import BeninSmallHolderCashews
from .bigearthnet import BigEarthNet, BigEarthNetV2
from .biomassters import BioMassters
from .bright import BRIGHTDFC2025
from .cabuar import CaBuAr
from .caffe import CaFFe
from .cbf import CanadianBuildingFootprints
from .cdl import CDL
from .chabud import ChaBuD
from .chesapeake import Chesapeake, ChesapeakeDC, ChesapeakeDE, ChesapeakeMD, ChesapeakeNY, ChesapeakePA, ChesapeakeVA, ChesapeakeWV, ChesapeakeCVPR
from .clay import ClayEmbeddings
from .cloud_cover import CloudCoverDetection
from .cms_mangrove_canopy import CMSGlobalMangroveCanopy
from .cowc import COWC, COWCCounting, COWCDetection
from .cropharvest import CropHarvest
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import TropicalCyclone
from .deepglobelandcover import DeepGlobeLandCover
from .dfc2022 import DFC2022
from .digital_typhoon import DigitalTyphoon
from .dior import DIOR
from .dl4gam import DL4GAMAlps
from .dlrsd import DLRSDBase, DLRSD, DLRSDMultilabel
from .dota import DOTA
from .earth_embeddings import EarthEmbeddings
from .earth_index import EarthIndexEmbeddings
from .eddmaps import EDDMapS
from .enmap import EnMAP
from .enviroatlas import EnviroAtlas
from .esd import ESDQuantizer, EmbeddedSeamlessData
from .esri2020 import Esri2020
from .etci2021 import ETCI2021
from .eudem import EUDEM
from .eurocrops import EuroCrops
from .everwatch import EverWatch
from .fair1m import FAIR1M
from .fire_risk import FireRisk
from .flair import FLAIRHUBBase, FLAIRHUB, FLAIRHUBToy
from .forestdamage import ForestDamage
from .ftw import FieldsOfTheWorld
from .gbif import GBIF
from .gbm import GlobalBuildingMap
from .geonrw import GeoNRW
from .gid15 import GID15
from .globalmangrovewatch import GlobalMangroveWatch
from .globbiomass import GlobBiomass
from .gse import GoogleSatelliteEmbedding
from .hyspecnet import HySpecNet11k
from .idtrees import IDTReeS
from .inaturalist import INaturalist
from .inria import InriaAerialImageLabeling
from .iobench import IOBench
from .l7irish import L7IrishImage, L7IrishMask, L7Irish
from .l8biome import L8BiomeImage, L8BiomeMask, L8Biome
from .landcoverai import LandCoverAIBase, LandCoverAIGeo, LandCoverAI, LandCoverAI100
from .landsat import Landsat, Landsat1, Landsat2, Landsat3, Landsat4MSS, Landsat4TM, Landsat5MSS, Landsat5TM, Landsat7, Landsat8, Landsat9
from .levircd import LEVIRCDBase, LEVIRCD, LEVIRCDPlus
from .loveda import LoveDA
from .major_tom import MajorTOMEmbeddings
from .mapinwild import MapInWild
from .mdas import MDAS
from .meta_chm import MetaCHM
from .millionaid import MillionAID
from .mmearth import MMEarth
from .mmflood import MMFloodComponent, MMFlood
from .naip import NAIP
from .nasa_marine_debris import NASAMarineDebris
from .nccm import NCCM
from .nlcd import NLCD
from .openaerialmap import TileUtils, OpenAerialMap
from .openbuildings import OpenBuildings
from .openstreetmap import OpenStreetMap
from .oscd import OSCD, OSCD100
from .pastis import PASTIS, PASTIS100
from .patternnet import PatternNet
from .potsdam import Potsdam2D
from .presto import PrestoEmbeddings
from .prisma import PRISMA
from .quakeset import QuakeSet
from .reforestree import ReforesTree
from .resisc45 import RESISC45
from .rwanda_field_boundary import RwandaFieldBoundary
from .s2_100k import S2100k
from .satlas import SatlasPretrain
from .seasonet import SeasoNet
from .seco import SeasonalContrastS2
from .sen12ms import SEN12MS
from .sentinel import Sentinel, Sentinel1, Sentinel2
from .skippd import SKIPPD
from .skyscript import SkyScript
from .so2sat import So2Sat
from .soda import SODAA
from .solar_plants_brazil import SolarPlantsBrazil
from .south_africa_crop_type import SouthAfricaCropType
from .south_america_soybean import SouthAmericaSoybean
from .ssl4eo import SSL4EO, SSL4EOL, SSL4EOS12
from .ssl4eo_benchmark import SSL4EOLBenchmark
from .substation import Substation
from .sustainbench_crop_yield import SustainBenchCropYield
from .tessera import TesseraEmbeddings
from .treesatai import TreeSatAI
from .ucmerced import UCMerced
from .usavars import USAVars
from .vaihingen import Vaihingen2D
from .vhr10 import VHR10
from .western_usa_live_fuel_moisture import WesternUSALiveFuelMoisture
from .worldstrat import WorldStrat
from .xbd import xBD, XView2, xBDDistShift
from .zuericrop import ZueriCrop

__all__ = (
    "ADVANCE",
    "AbovegroundLiveWoodyBiomassDensity",
    "AgriFieldNet",
    "AgriFieldNetImage",
    "AgriFieldNetMask",
    "AirQuality",
    "Airphen",
    "AsterGDEM",
    "BRIGHTDFC2025",
    "BeninSmallHolderCashews",
    "BigEarthNet",
    "BigEarthNetV2",
    "BioMassters",
    "BoundingBox",
    "CDL",
    "CMSGlobalMangroveCanopy",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "CV4AKenyaCropType",
    "CaBuAr",
    "CaFFe",
    "CanadianBuildingFootprints",
    "ChaBuD",
    "Chesapeake",
    "ChesapeakeCVPR",
    "ChesapeakeDC",
    "ChesapeakeDE",
    "ChesapeakeMD",
    "ChesapeakeNY",
    "ChesapeakePA",
    "ChesapeakeVA",
    "ChesapeakeWV",
    "ClayEmbeddings",
    "CloudCoverDetection",
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
    "CropHarvest",
    "DFC2022",
    "DIOR",
    "DL4GAMAlps",
    "DLRSD",
    "DLRSDBase",
    "DLRSDMultilabel",
    "DOTA",
    "DatasetNotFoundError",
    "DeepGlobeLandCover",
    "DependencyNotFoundError",
    "DigitalTyphoon",
    "EDDMapS",
    "ESDQuantizer",
    "ETCI2021",
    "EUDEM",
    "EarthEmbeddings",
    "EarthIndexEmbeddings",
    "EmbeddedSeamlessData",
    "EnMAP",
    "EnviroAtlas",
    "Esri2020",
    "EuroCrops",
    "EuroSAT",
    "EuroSAT100",
    "EuroSATSpatial",
    "EverWatch",
    "FAIR1M",
    "FLAIRHUB",
    "FLAIRHUBBase",
    "FLAIRHUBToy",
    "FieldsOfTheWorld",
    "FireRisk",
    "ForestDamage",
    "GBIF",
    "GID15",
    "GeoDataset",
    "GeoNRW",
    "GlobBiomass",
    "GlobalBuildingMap",
    "GlobalMangroveWatch",
    "GoogleSatelliteEmbedding",
    "HySpecNet11k",
    "IDTReeS",
    "INaturalist",
    "IOBench",
    "InriaAerialImageLabeling",
    "IntersectionDataset",
    "L7Irish",
    "L7IrishImage",
    "L7IrishMask",
    "L8Biome",
    "L8BiomeImage",
    "L8BiomeMask",
    "LEVIRCD",
    "LEVIRCDBase",
    "LEVIRCDPlus",
    "LandCoverAI",
    "LandCoverAI100",
    "LandCoverAIBase",
    "LandCoverAIGeo",
    "Landsat",
    "Landsat1",
    "Landsat2",
    "Landsat3",
    "Landsat4MSS",
    "Landsat4TM",
    "Landsat5MSS",
    "Landsat5TM",
    "Landsat7",
    "Landsat8",
    "Landsat9",
    "LoveDA",
    "MDAS",
    "MMEarth",
    "MMFlood",
    "MMFloodComponent",
    "MajorTOMEmbeddings",
    "MapInWild",
    "MetaCHM",
    "MillionAID",
    "NAIP",
    "NASAMarineDebris",
    "NCCM",
    "NLCD",
    "NonGeoClassificationDataset",
    "NonGeoDataset",
    "OSCD",
    "OSCD100",
    "OpenAerialMap",
    "OpenBuildings",
    "OpenStreetMap",
    "PASTIS",
    "PASTIS100",
    "PRISMA",
    "PatternNet",
    "PlottingMixin",
    "Potsdam2D",
    "PrestoEmbeddings",
    "QuakeSet",
    "RESISC45",
    "RGBBandsMissingError",
    "RasterDataset",
    "ReforesTree",
    "RwandaFieldBoundary",
    "S2100k",
    "SEN12MS",
    "SKIPPD",
    "SODAA",
    "SSL4EO",
    "SSL4EOL",
    "SSL4EOLBenchmark",
    "SSL4EOS12",
    "SatlasPretrain",
    "SeasoNet",
    "SeasonalContrastS2",
    "Sentinel",
    "Sentinel1",
    "Sentinel2",
    "SkyScript",
    "So2Sat",
    "SolarPlantsBrazil",
    "SouthAfricaCropType",
    "SouthAmericaSoybean",
    "SpaceNet",
    "SpaceNet1",
    "SpaceNet2",
    "SpaceNet3",
    "SpaceNet4",
    "SpaceNet5",
    "SpaceNet6",
    "SpaceNet7",
    "SpaceNet8",
    "Substation",
    "SustainBenchCropYield",
    "TesseraEmbeddings",
    "TileUtils",
    "TreeSatAI",
    "TropicalCyclone",
    "UCMerced",
    "USAVars",
    "UnionDataset",
    "VHR10",
    "Vaihingen2D",
    "VectorDataset",
    "WesternUSALiveFuelMoisture",
    "WorldStrat",
    "XView2",
    "XarrayDataset",
    "ZueriCrop",
    "concat_samples",
    "merge_samples",
    "random_bbox_assignment",
    "random_bbox_splitting",
    "random_grid_cell_assignment",
    "roi_split",
    "stack_samples",
    "time_series_split",
    "unbind_samples",
    "xBD",
    "xBDDistShift",
)
