import json
import os
import zipfile

import pytest

from keras_climate.datasets import DatasetNotFoundError, IntersectionDataset, UnionDataset
from keras_climate.datasets.cbf import CanadianBuildingFootprints


def _write_geojson(path):
    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
                    ],
                },
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(geojson, f)


@pytest.fixture
def prepared_root(tmp_path, monkeypatch):
    monkeypatch.setattr(CanadianBuildingFootprints, "provinces_territories", ("Alberta",))
    monkeypatch.setattr(CanadianBuildingFootprints, "md5s", (None,))
    root = str(tmp_path)
    geojson_path = os.path.join(root, "Alberta.geojson")
    _write_geojson(geojson_path)
    # _check_integrity() requires the archive to be present too (the dataset
    # remembers what it "downloaded"); the loose .geojson is what VectorDataset
    # actually reads.
    with zipfile.ZipFile(os.path.join(root, "Alberta.zip"), "w") as zf:
        zf.write(geojson_path, arcname="Alberta.geojson")
    return root


def test_getitem(prepared_root):
    ds = CanadianBuildingFootprints(prepared_root, res=(0.1, 0.1), checksum=False)
    x = ds[ds.bounds]
    assert "mask" in x


def test_len(prepared_root):
    ds = CanadianBuildingFootprints(prepared_root, res=(0.1, 0.1), checksum=False)
    assert len(ds) == 1


def test_and(prepared_root):
    ds = CanadianBuildingFootprints(prepared_root, res=(0.1, 0.1), checksum=False)
    combo = ds & ds
    assert isinstance(combo, IntersectionDataset)


def test_or(prepared_root):
    ds = CanadianBuildingFootprints(prepared_root, res=(0.1, 0.1), checksum=False)
    combo = ds | ds
    assert isinstance(combo, UnionDataset)


def test_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(CanadianBuildingFootprints, "provinces_territories", ("Alberta",))
    monkeypatch.setattr(CanadianBuildingFootprints, "md5s", (None,))
    with pytest.raises(DatasetNotFoundError):
        CanadianBuildingFootprints(str(tmp_path), download=False, checksum=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = CanadianBuildingFootprints(prepared_root, res=(0.1, 0.1), checksum=False)
    x = ds[ds.bounds]
    ds.plot(x, suptitle="Test")
    plt.close()
    x["prediction"] = x["mask"]
    ds.plot(x, suptitle="Prediction")
    plt.close()
