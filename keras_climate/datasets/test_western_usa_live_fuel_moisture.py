import json
import os

import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.western_usa_live_fuel_moisture import WesternUSALiveFuelMoisture


def _feature():
    properties = {"percent(t)": 132.6666667, "site": "Blackstone", "date": "6/30/15"}
    for name in WesternUSALiveFuelMoisture.all_variable_names:
        if name not in ("lat", "lon"):
            properties[name] = 1.0
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Point", "coordinates": [-115.8855556, 42.44111111]},
    }


@pytest.fixture
def prepared_root(tmp_path):
    data_dir = os.path.join(str(tmp_path), "su_sar_moisture_content")
    os.makedirs(data_dir)
    for i in range(1, 4):
        with open(os.path.join(data_dir, f"feature_{i:04}.geojson"), "w") as f:
            json.dump(_feature(), f)
    return str(tmp_path)


def test_getitem(prepared_root):
    ds = WesternUSALiveFuelMoisture(root=prepared_root, download=False)
    sample = ds[0]
    assert "input" in sample
    assert "label" in sample
    assert tuple(sample["input"].shape) == (len(WesternUSALiveFuelMoisture.all_variable_names),)


def test_len(prepared_root):
    ds = WesternUSALiveFuelMoisture(root=prepared_root, download=False)
    assert len(ds) == 3


def test_invalid_input_features(prepared_root):
    with pytest.raises(AssertionError):
        WesternUSALiveFuelMoisture(root=prepared_root, input_features=("not_a_feature",))


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        WesternUSALiveFuelMoisture(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = WesternUSALiveFuelMoisture(root=prepared_root, download=False)
    sample = ds[0]
    ds.plot(sample, variables_to_plot=["vv"])
    plt.close()
    ds.plot(sample, show_titles=False, suptitle="Custom title")
    plt.close()
