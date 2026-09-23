import csv
import gzip
import json
import os
import shutil

import pytest
from shapely import Polygon

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.openbuildings import OpenBuildings

SIZE = 0.05


def _csv_row(lat, long):
    width, height = SIZE / 10, SIZE / 10
    minx = long - 0.5 * width
    maxx = long + 0.5 * width
    miny = lat - 0.5 * height
    maxy = lat + 0.5 * height
    coords = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]
    polygon = Polygon(coords)
    return {
        "latitude": lat,
        "longitude": long,
        "area_in_meters": 1.0,
        "confidence": 1.0,
        "geometry": polygon.wkt,
        "full_plus_code": "ABC",
    }


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    csvname = os.path.join(root, "000_buildings.csv")
    zipfilename = csvname + ".gz"

    meta_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [0.0, SIZE], [SIZE, SIZE], [SIZE, 0.0], [0.0, 0.0]]
                    ],
                },
                "properties": {
                    "tile_id": "000",
                    "tile_url": "polygons_s2_level_4_gzip/000_buildings.csv.gz",
                    "size_mb": 0.2,
                },
            }
        ],
    }
    with open(os.path.join(root, "tiles.geojson"), "w") as fp:
        json.dump(meta_data, fp)

    fourth = SIZE / 4
    rows = [_csv_row(fourth, fourth), _csv_row(SIZE - fourth, SIZE - fourth)]
    with open(csvname, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    with open(csvname, "rb") as f_in, gzip.open(zipfilename, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(csvname)

    return root


def test_getitem(prepared_root):
    ds = OpenBuildings(paths=prepared_root, checksum=False)
    sample = ds[ds.bounds]
    assert "mask" in sample
    assert sample["mask"].shape[-1] == 1


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        OpenBuildings(paths=str(tmp_path), checksum=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = OpenBuildings(paths=prepared_root, checksum=False)
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()
    sample["prediction"] = sample["mask"]
    ds.plot(sample)
    plt.close()
