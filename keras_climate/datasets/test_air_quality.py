import os

import pandas as pd
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.air_quality import AirQuality


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    rows = []
    for i in range(20):
        rows.append(
            {
                "Date": "15/01/2004",
                "Time": f"{i % 24:02d}:00:00",
                "CO(GT)": 1.0 + i,
                "NMHC(GT)": 2.0 + i,
                "C6H6(GT)": 3.0 + i,
                "NOx(GT)": 4.0 + i,
                "NO2(GT)": 5.0 + i,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, AirQuality.data_file_name), index=False)
    return root


def test_getitem(prepared_root):
    ds = AirQuality(prepared_root, input_steps=3, target_steps=1)
    item = ds[0]
    assert tuple(item["input"].shape)[0] == 3
    assert tuple(item["target"].shape)[0] == 1
    assert item["input"].dtype == "float32"


def test_len(prepared_root):
    ds = AirQuality(prepared_root, input_steps=3, target_steps=1)
    assert len(ds) == 20 - 3 - 1 + 1


def test_feature_subset(prepared_root):
    features = ["CO(GT)", "NMHC(GT)"]
    ds = AirQuality(
        prepared_root, input_features=features, target_features=features
    )
    item = ds[0]
    # Date/Time are not in usecols, so no cyclical features are added
    assert tuple(item["input"].shape)[1] == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        AirQuality(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = AirQuality(prepared_root)
    sample = ds[0]
    sample["prediction"] = sample["target"]
    ds.plot(sample, suptitle="Test")
    plt.close()
    ds.plot(sample, show_titles=False)
    plt.close()
