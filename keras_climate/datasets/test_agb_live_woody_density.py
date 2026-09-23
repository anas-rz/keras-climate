import json
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.agb_live_woody_density import (
    AbovegroundLiveWoodyBiomassDensity,
)


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    tile_id = "00N_000E"
    tif_name = f"{tile_id}.tif"

    transform = from_origin(0, 8, 1, 1)
    data = np.random.randint(0, 500, (1, 8, 8)).astype("uint16")
    with rasterio.open(
        os.path.join(root, tif_name),
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "properties": {
                    "tile_id": tile_id,
                    "Mg_px_1_download": os.path.join(root, tif_name),
                }
            }
        ],
    }
    with open(
        os.path.join(root, AbovegroundLiveWoodyBiomassDensity.base_filename), "w"
    ) as f:
        json.dump(geojson, f)

    return root


def test_getitem(prepared_root):
    ds = AbovegroundLiveWoodyBiomassDensity(prepared_root)
    x = ds[ds.bounds]
    assert "mask" in x


def test_len(prepared_root):
    ds = AbovegroundLiveWoodyBiomassDensity(prepared_root)
    assert len(ds) == 1


def test_not_found(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        AbovegroundLiveWoodyBiomassDensity(str(tmp_path))


def test_and(prepared_root):
    ds = AbovegroundLiveWoodyBiomassDensity(prepared_root)
    assert isinstance(ds & ds, IntersectionDataset)


def test_or(prepared_root):
    ds = AbovegroundLiveWoodyBiomassDensity(prepared_root)
    assert isinstance(ds | ds, UnionDataset)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = AbovegroundLiveWoodyBiomassDensity(prepared_root)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()


def test_plot_prediction(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from keras import ops

    ds = AbovegroundLiveWoodyBiomassDensity(prepared_root)
    x = ds[ds.bounds]
    x["prediction"] = ops.convert_to_tensor(x["mask"])
    ds.plot(x, suptitle="Prediction")
    plt.close()
