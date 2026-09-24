# Datasets API

Auto-generated from source — a multi-backend Keras port of
[`torchgeo.datasets`](https://torchgeo.readthedocs.io/en/stable/api/datasets.html),
covering the base geospatial/non-geospatial dataset classes plus the full
catalog of concrete leaf datasets (EuroSAT, So2Sat, Sentinel, SpaceNet,
Copernicus-Bench, ...). See [Data](../models/data.md) for a narrative guide
to loading, sampling, and transforming data.

Image/mask samples use the channels-last layout (`H x W x C`), a deliberate
deviation from torchgeo's channels-first convention to match the rest of
`keras_climate`.

## Core Framework

::: keras_climate.datasets.errors.DatasetNotFoundError

::: keras_climate.datasets.errors.DependencyNotFoundError

::: keras_climate.datasets.errors.RGBBandsMissingError

::: keras_climate.datasets.geo.GeoDataset

::: keras_climate.datasets.geo.IntersectionDataset

::: keras_climate.datasets.geo.NonGeoClassificationDataset

::: keras_climate.datasets.geo.NonGeoDataset

::: keras_climate.datasets.geo.RasterDataset

::: keras_climate.datasets.geo.UnionDataset

::: keras_climate.datasets.geo.VectorDataset

::: keras_climate.datasets.geo.XarrayDataset

::: keras_climate.datasets.mixins.PlottingMixin

::: keras_climate.datasets.splits.random_bbox_assignment

::: keras_climate.datasets.splits.random_bbox_splitting

::: keras_climate.datasets.splits.random_grid_cell_assignment

::: keras_climate.datasets.splits.roi_split

::: keras_climate.datasets.splits.time_series_split

::: keras_climate.datasets.utils.BoundingBox

::: keras_climate.datasets.utils.concat_samples

::: keras_climate.datasets.utils.merge_samples

::: keras_climate.datasets.utils.stack_samples

::: keras_climate.datasets.utils.unbind_samples

## Copernicus-Bench

::: keras_climate.datasets.copernicus.CopernicusBenchAQNO2S5P

::: keras_climate.datasets.copernicus.CopernicusBenchAQO3S5P

::: keras_climate.datasets.copernicus.CopernicusBenchBase

::: keras_climate.datasets.copernicus.CopernicusBenchBigEarthNetS1

::: keras_climate.datasets.copernicus.CopernicusBenchBigEarthNetS2

::: keras_climate.datasets.copernicus.CopernicusBenchBiomassS3

::: keras_climate.datasets.copernicus.CopernicusBenchCloudS2

::: keras_climate.datasets.copernicus.CopernicusBenchCloudS3

::: keras_climate.datasets.copernicus.CopernicusBenchDFC2020S1

::: keras_climate.datasets.copernicus.CopernicusBenchDFC2020S2

::: keras_climate.datasets.copernicus.CopernicusBenchEuroSATS1

::: keras_climate.datasets.copernicus.CopernicusBenchEuroSATS2

::: keras_climate.datasets.copernicus.CopernicusBenchFloodS1

::: keras_climate.datasets.copernicus.CopernicusBenchLC100ClsS3

::: keras_climate.datasets.copernicus.CopernicusBenchLC100SegS3

::: keras_climate.datasets.copernicus.CopernicusBenchLCZS2

::: keras_climate.datasets.copernicus.CopernicusEmbed

::: keras_climate.datasets.copernicus.CopernicusPretrain

## SpaceNet

::: keras_climate.datasets.spacenet.SpaceNet

::: keras_climate.datasets.spacenet.SpaceNet1

::: keras_climate.datasets.spacenet.SpaceNet2

::: keras_climate.datasets.spacenet.SpaceNet3

::: keras_climate.datasets.spacenet.SpaceNet4

::: keras_climate.datasets.spacenet.SpaceNet5

::: keras_climate.datasets.spacenet.SpaceNet6

::: keras_climate.datasets.spacenet.SpaceNet7

::: keras_climate.datasets.spacenet.SpaceNet8

## Dataset Catalog

::: keras_climate.datasets.advance.ADVANCE

::: keras_climate.datasets.agb_live_woody_density.AbovegroundLiveWoodyBiomassDensity

::: keras_climate.datasets.agrifieldnet.AgriFieldNet

::: keras_climate.datasets.agrifieldnet.AgriFieldNetImage

::: keras_climate.datasets.agrifieldnet.AgriFieldNetMask

::: keras_climate.datasets.air_quality.AirQuality

::: keras_climate.datasets.airphen.Airphen

::: keras_climate.datasets.astergdem.AsterGDEM

::: keras_climate.datasets.benin_cashews.BeninSmallHolderCashews

::: keras_climate.datasets.bigearthnet.BigEarthNet

::: keras_climate.datasets.bigearthnet.BigEarthNetV2

::: keras_climate.datasets.biomassters.BioMassters

::: keras_climate.datasets.bright.BRIGHTDFC2025

::: keras_climate.datasets.cabuar.CaBuAr

::: keras_climate.datasets.caffe.CaFFe

::: keras_climate.datasets.cbf.CanadianBuildingFootprints

::: keras_climate.datasets.cdl.CDL

::: keras_climate.datasets.chabud.ChaBuD

::: keras_climate.datasets.chesapeake.Chesapeake

::: keras_climate.datasets.chesapeake.ChesapeakeCVPR

::: keras_climate.datasets.chesapeake.ChesapeakeDC

::: keras_climate.datasets.chesapeake.ChesapeakeDE

::: keras_climate.datasets.chesapeake.ChesapeakeMD

::: keras_climate.datasets.chesapeake.ChesapeakeNY

::: keras_climate.datasets.chesapeake.ChesapeakePA

::: keras_climate.datasets.chesapeake.ChesapeakeVA

::: keras_climate.datasets.chesapeake.ChesapeakeWV

::: keras_climate.datasets.clay.ClayEmbeddings

::: keras_climate.datasets.cloud_cover.CloudCoverDetection

::: keras_climate.datasets.cms_mangrove_canopy.CMSGlobalMangroveCanopy

::: keras_climate.datasets.cowc.COWC

::: keras_climate.datasets.cowc.COWCCounting

::: keras_climate.datasets.cowc.COWCDetection

::: keras_climate.datasets.cropharvest.CropHarvest

::: keras_climate.datasets.cv4a_kenya_crop_type.CV4AKenyaCropType

::: keras_climate.datasets.cyclone.TropicalCyclone

::: keras_climate.datasets.deepglobelandcover.DeepGlobeLandCover

::: keras_climate.datasets.dfc2022.DFC2022

::: keras_climate.datasets.digital_typhoon.DigitalTyphoon

::: keras_climate.datasets.dior.DIOR

::: keras_climate.datasets.dl4gam.DL4GAMAlps

::: keras_climate.datasets.dlrsd.DLRSD

::: keras_climate.datasets.dlrsd.DLRSDBase

::: keras_climate.datasets.dlrsd.DLRSDMultilabel

::: keras_climate.datasets.dota.DOTA

::: keras_climate.datasets.earth_embeddings.EarthEmbeddings

::: keras_climate.datasets.earth_index.EarthIndexEmbeddings

::: keras_climate.datasets.eddmaps.EDDMapS

::: keras_climate.datasets.enmap.EnMAP

::: keras_climate.datasets.enviroatlas.EnviroAtlas

::: keras_climate.datasets.esd.ESDQuantizer

::: keras_climate.datasets.esd.EmbeddedSeamlessData

::: keras_climate.datasets.esri2020.Esri2020

::: keras_climate.datasets.etci2021.ETCI2021

::: keras_climate.datasets.eudem.EUDEM

::: keras_climate.datasets.eurocrops.EuroCrops

::: keras_climate.datasets.eurosat.EuroSAT

::: keras_climate.datasets.eurosat.EuroSAT100

::: keras_climate.datasets.eurosat.EuroSATSpatial

::: keras_climate.datasets.everwatch.EverWatch

::: keras_climate.datasets.fair1m.FAIR1M

::: keras_climate.datasets.fire_risk.FireRisk

::: keras_climate.datasets.flair.FLAIRHUB

::: keras_climate.datasets.flair.FLAIRHUBBase

::: keras_climate.datasets.flair.FLAIRHUBToy

::: keras_climate.datasets.forestdamage.ForestDamage

::: keras_climate.datasets.ftw.FieldsOfTheWorld

::: keras_climate.datasets.gbif.GBIF

::: keras_climate.datasets.gbm.GlobalBuildingMap

::: keras_climate.datasets.geonrw.GeoNRW

::: keras_climate.datasets.gid15.GID15

::: keras_climate.datasets.globalmangrovewatch.GlobalMangroveWatch

::: keras_climate.datasets.globbiomass.GlobBiomass

::: keras_climate.datasets.gse.GoogleSatelliteEmbedding

::: keras_climate.datasets.hyspecnet.HySpecNet11k

::: keras_climate.datasets.idtrees.IDTReeS

::: keras_climate.datasets.inaturalist.INaturalist

::: keras_climate.datasets.inria.InriaAerialImageLabeling

::: keras_climate.datasets.iobench.IOBench

::: keras_climate.datasets.l7irish.L7Irish

::: keras_climate.datasets.l7irish.L7IrishImage

::: keras_climate.datasets.l7irish.L7IrishMask

::: keras_climate.datasets.l8biome.L8Biome

::: keras_climate.datasets.l8biome.L8BiomeImage

::: keras_climate.datasets.l8biome.L8BiomeMask

::: keras_climate.datasets.landcoverai.LandCoverAI

::: keras_climate.datasets.landcoverai.LandCoverAI100

::: keras_climate.datasets.landcoverai.LandCoverAIBase

::: keras_climate.datasets.landcoverai.LandCoverAIGeo

::: keras_climate.datasets.landsat.Landsat

::: keras_climate.datasets.landsat.Landsat1

::: keras_climate.datasets.landsat.Landsat2

::: keras_climate.datasets.landsat.Landsat3

::: keras_climate.datasets.landsat.Landsat4MSS

::: keras_climate.datasets.landsat.Landsat4TM

::: keras_climate.datasets.landsat.Landsat5MSS

::: keras_climate.datasets.landsat.Landsat5TM

::: keras_climate.datasets.landsat.Landsat7

::: keras_climate.datasets.landsat.Landsat8

::: keras_climate.datasets.landsat.Landsat9

::: keras_climate.datasets.levircd.LEVIRCD

::: keras_climate.datasets.levircd.LEVIRCDBase

::: keras_climate.datasets.levircd.LEVIRCDPlus

::: keras_climate.datasets.loveda.LoveDA

::: keras_climate.datasets.major_tom.MajorTOMEmbeddings

::: keras_climate.datasets.mapinwild.MapInWild

::: keras_climate.datasets.mdas.MDAS

::: keras_climate.datasets.meta_chm.MetaCHM

::: keras_climate.datasets.millionaid.MillionAID

::: keras_climate.datasets.mmearth.MMEarth

::: keras_climate.datasets.mmflood.MMFlood

::: keras_climate.datasets.mmflood.MMFloodComponent

::: keras_climate.datasets.naip.NAIP

::: keras_climate.datasets.nasa_marine_debris.NASAMarineDebris

::: keras_climate.datasets.nccm.NCCM

::: keras_climate.datasets.nlcd.NLCD

::: keras_climate.datasets.openaerialmap.OpenAerialMap

::: keras_climate.datasets.openaerialmap.TileUtils

::: keras_climate.datasets.openbuildings.OpenBuildings

::: keras_climate.datasets.openstreetmap.OpenStreetMap

::: keras_climate.datasets.oscd.OSCD

::: keras_climate.datasets.oscd.OSCD100

::: keras_climate.datasets.pastis.PASTIS

::: keras_climate.datasets.pastis.PASTIS100

::: keras_climate.datasets.patternnet.PatternNet

::: keras_climate.datasets.potsdam.Potsdam2D

::: keras_climate.datasets.presto.PrestoEmbeddings

::: keras_climate.datasets.prisma.PRISMA

::: keras_climate.datasets.quakeset.QuakeSet

::: keras_climate.datasets.reforestree.ReforesTree

::: keras_climate.datasets.resisc45.RESISC45

::: keras_climate.datasets.rwanda_field_boundary.RwandaFieldBoundary

::: keras_climate.datasets.s2_100k.S2100k

::: keras_climate.datasets.satlas.SatlasPretrain

::: keras_climate.datasets.seasonet.SeasoNet

::: keras_climate.datasets.seco.SeasonalContrastS2

::: keras_climate.datasets.sen12ms.SEN12MS

::: keras_climate.datasets.sentinel.Sentinel

::: keras_climate.datasets.sentinel.Sentinel1

::: keras_climate.datasets.sentinel.Sentinel2

::: keras_climate.datasets.skippd.SKIPPD

::: keras_climate.datasets.skyscript.SkyScript

::: keras_climate.datasets.so2sat.So2Sat

::: keras_climate.datasets.soda.SODAA

::: keras_climate.datasets.solar_plants_brazil.SolarPlantsBrazil

::: keras_climate.datasets.south_africa_crop_type.SouthAfricaCropType

::: keras_climate.datasets.south_america_soybean.SouthAmericaSoybean

::: keras_climate.datasets.ssl4eo.SSL4EO

::: keras_climate.datasets.ssl4eo.SSL4EOL

::: keras_climate.datasets.ssl4eo.SSL4EOS12

::: keras_climate.datasets.ssl4eo_benchmark.SSL4EOLBenchmark

::: keras_climate.datasets.substation.Substation

::: keras_climate.datasets.sustainbench_crop_yield.SustainBenchCropYield

::: keras_climate.datasets.tessera.TesseraEmbeddings

::: keras_climate.datasets.treesatai.TreeSatAI

::: keras_climate.datasets.ucmerced.UCMerced

::: keras_climate.datasets.usavars.USAVars

::: keras_climate.datasets.vaihingen.Vaihingen2D

::: keras_climate.datasets.vhr10.VHR10

::: keras_climate.datasets.western_usa_live_fuel_moisture.WesternUSALiveFuelMoisture

::: keras_climate.datasets.worldstrat.WorldStrat

::: keras_climate.datasets.xbd.XView2

::: keras_climate.datasets.xbd.xBD

::: keras_climate.datasets.xbd.xBDDistShift

::: keras_climate.datasets.zuericrop.ZueriCrop

