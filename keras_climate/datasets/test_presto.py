import numpy as np
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.presto import PrestoEmbeddings


def _write_embeddings_tif(path, size=4, bands=128):
    transform = from_origin(0, size, 1, 1)
    data = np.random.rand(bands, size, size).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def test_getitem(tmp_path):
    _write_embeddings_tif(tmp_path / "Togo_Presto_embeddings_v2025_06_19_00000-00001.tif")
    ds = PrestoEmbeddings(paths=str(tmp_path))
    x, y, t = ds.bounds
    sample = ds[x, y, t]
    assert tuple(sample["image"].shape) == (4, 4, 128)
    assert "bounds" in sample


def test_len(tmp_path):
    _write_embeddings_tif(tmp_path / "Togo_Presto_embeddings_v2025_06_19_00000-00001.tif")
    ds = PrestoEmbeddings(paths=str(tmp_path))
    assert len(ds) == 1


def test_plot(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _write_embeddings_tif(tmp_path / "Togo_Presto_embeddings_v2025_06_19_00000-00001.tif")
    ds = PrestoEmbeddings(paths=str(tmp_path))
    x, y, t = ds.bounds
    sample = ds[x, y, t]
    ds.plot(sample, suptitle="Test")
    plt.close()
