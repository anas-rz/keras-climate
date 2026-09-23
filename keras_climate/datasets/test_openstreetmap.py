import hashlib
import json
import os

import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.openstreetmap import OpenStreetMap

BBOX = (2.3520, 48.8565, 2.3525, 48.8570)
CLASSES = [{"name": "buildings", "selector": [{"building": "*"}]}]

BUILDING_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "building": "residential",
                "osm_id": 12345,
                "osm_type": "way",
                "label": 1,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [2.3522, 48.8566],
                        [2.3524, 48.8566],
                        [2.3524, 48.8568],
                        [2.3522, 48.8568],
                        [2.3522, 48.8566],
                    ]
                ],
            },
        }
    ],
}


def _cache_filename(root, bbox, classes):
    cache_key = {"bbox": list(bbox), "classes": classes}
    cache_str = json.dumps(cache_key, sort_keys=True)
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:16]
    return os.path.join(root, f"osm_features_{cache_hash}.geojson")


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    filename = _cache_filename(root, BBOX, CLASSES)
    with open(filename, "w") as f:
        json.dump(BUILDING_GEOJSON, f)
    return root


def test_getitem(prepared_root):
    ds = OpenStreetMap(bbox=BBOX, classes=CLASSES, paths=prepared_root)
    sample = ds[ds.bounds]
    assert "mask" in sample


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        OpenStreetMap(bbox=BBOX, classes=CLASSES, paths=str(tmp_path), download=False)


def test_invalid_classes_type():
    with pytest.raises(TypeError):
        OpenStreetMap(bbox=BBOX, classes="bogus")


def test_invalid_classes_empty():
    with pytest.raises(TypeError):
        OpenStreetMap(bbox=BBOX, classes=[])


def test_invalid_class_missing_keys():
    with pytest.raises(ValueError):
        OpenStreetMap(bbox=BBOX, classes=[{"name": "buildings"}])


def test_invalid_selector_type():
    with pytest.raises(TypeError):
        OpenStreetMap(bbox=BBOX, classes=[{"name": "buildings", "selector": "bogus"}])


def test_feature_matches_selector(prepared_root):
    ds = OpenStreetMap(bbox=BBOX, classes=CLASSES, paths=prepared_root)
    assert ds._feature_matches_selector({"building": "yes"}, {"building": "*"})
    assert not ds._feature_matches_selector({}, {"building": "*"})
    assert ds._feature_matches_selector(
        {"highway": "primary"}, {"highway": ["primary", "secondary"]}
    )


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = OpenStreetMap(bbox=BBOX, classes=CLASSES, paths=prepared_root)
    sample = ds[ds.bounds]
    ds.plot(sample, suptitle="Test")
    plt.close()
