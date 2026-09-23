import os

import h5py
import numpy as np
import pandas as pd
import pytest

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets.digital_typhoon import DigitalTyphoon

IMAGE_SIZE = 4
NUM_TYPHOON_IDS = 2
NUM_IMAGES_PER_ID = 4


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    data_root = os.path.join(root, "WP")
    os.makedirs(os.path.join(data_root, "image"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "metadata"), exist_ok=True)

    rng = np.random.RandomState(0)
    all_dfs = []
    for typhoon_id in range(NUM_TYPHOON_IDS):
        os.makedirs(os.path.join(data_root, "image", str(typhoon_id)), exist_ok=True)

        image_paths = []
        for image_id in range(NUM_IMAGES_PER_ID):
            fname = f"{image_id}.h5"
            with h5py.File(os.path.join(data_root, "image", str(typhoon_id), fname), "w") as hf:
                hf.create_dataset("Infrared", data=rng.uniform(180, 290, (IMAGE_SIZE, IMAGE_SIZE)))
            image_paths.append(fname)

        times = pd.date_range(start="2000-01-01", periods=NUM_IMAGES_PER_ID, freq="h")
        df = pd.DataFrame(
            {
                "id": np.repeat(typhoon_id, NUM_IMAGES_PER_ID),
                "image_path": image_paths,
                "year": times.year,
                "month": times.month,
                "day": times.day,
                "hour": times.hour,
                "grade": rng.randint(1, 5, NUM_IMAGES_PER_ID),
                "lat": rng.uniform(-90, 90, NUM_IMAGES_PER_ID),
                "lng": rng.uniform(-180, 180, NUM_IMAGES_PER_ID),
                "pressure": rng.uniform(900, 1000, NUM_IMAGES_PER_ID),
                "wind": rng.uniform(0, 100, NUM_IMAGES_PER_ID),
                "dir50": rng.randint(0, 360, NUM_IMAGES_PER_ID),
                "long50": rng.randint(0, 100, NUM_IMAGES_PER_ID),
                "short50": rng.randint(0, 100, NUM_IMAGES_PER_ID),
                "dir30": rng.randint(0, 360, NUM_IMAGES_PER_ID),
                "long30": rng.randint(0, 100, NUM_IMAGES_PER_ID),
                "short30": rng.randint(0, 100, NUM_IMAGES_PER_ID),
                "landfall": rng.randint(0, 2, NUM_IMAGES_PER_ID),
                "intp": rng.randint(0, 2, NUM_IMAGES_PER_ID),
                "file_1": [f"{idx}.h5" for idx in range(NUM_IMAGES_PER_ID)],
            }
        )
        df.to_csv(os.path.join(data_root, "metadata", f"{typhoon_id}.csv"), index=False)
        all_dfs.append(df)

    aux_data = pd.concat(all_dfs)
    aux_data.to_csv(os.path.join(data_root, "aux_data.csv"), index=False)

    return root


def test_getitem(prepared_root):
    ds = DigitalTyphoon(root=prepared_root, sequence_length=2, download=False)
    sample = ds[0]
    assert tuple(sample["image"].shape) == (IMAGE_SIZE, IMAGE_SIZE, 2)
    assert "label" in sample
    assert tuple(sample["label"].shape) == (1,)


def test_len(prepared_root):
    ds = DigitalTyphoon(root=prepared_root, sequence_length=2, download=False)
    assert len(ds) > 0


def test_invalid_task():
    with pytest.raises(AssertionError):
        DigitalTyphoon(task="bad")


def test_invalid_features():
    with pytest.raises(AssertionError):
        DigitalTyphoon(features=["not_a_feature"])


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        DigitalTyphoon(root=str(tmp_path), download=False)


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = DigitalTyphoon(root=prepared_root, sequence_length=2, download=False)
    ds.plot(ds[0], suptitle="Test")
    plt.close()
