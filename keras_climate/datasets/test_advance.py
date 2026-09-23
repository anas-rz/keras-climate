import os
import wave

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("scipy")

from keras_climate.datasets import DatasetNotFoundError
from keras_climate.datasets._test_helpers import assert_dtype
from keras_climate.datasets.advance import ADVANCE


def _write_wav(path):
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(np.zeros(100, dtype="int16").tobytes())


@pytest.fixture
def prepared_root(tmp_path):
    root = str(tmp_path)
    for cls in ("airport", "beach"):
        vision_dir = os.path.join(root, "vision", cls)
        sound_dir = os.path.join(root, "sound", cls)
        os.makedirs(vision_dir, exist_ok=True)
        os.makedirs(sound_dir, exist_ok=True)
        img = Image.fromarray(
            np.random.randint(0, 255, (8, 8, 3), dtype="uint8"), mode="RGB"
        )
        img.save(os.path.join(vision_dir, "0.jpg"))
        _write_wav(os.path.join(sound_dir, "0.wav"))
    return root


def test_getitem(prepared_root):
    ds = ADVANCE(prepared_root)
    x = ds[0]
    assert tuple(x["image"].shape) == (8, 8, 3)
    assert_dtype(x["image"], "float32")
    assert "audio" in x
    assert "label" in x


def test_len(prepared_root):
    ds = ADVANCE(prepared_root)
    assert len(ds) == 2


def test_not_downloaded(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        ADVANCE(str(tmp_path))


def test_plot(prepared_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = ADVANCE(prepared_root)
    x = ds[0]
    ds.plot(x, suptitle="Test")
    plt.close()

    x["prediction"] = x["label"]
    ds.plot(x, show_titles=False)
    plt.close()
