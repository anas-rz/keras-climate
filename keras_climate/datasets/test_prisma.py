import numpy as np
import rasterio
from rasterio.transform import from_origin

from keras_climate.datasets.prisma import PRISMA


def _write_prisma_tif(path, size=8, bands=40):
    transform = from_origin(0, size, 1, 1)
    data = np.random.randint(0, 1000, (bands, size, size)).astype("uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=bands,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def test_getitem(tmp_path):
    _write_prisma_tif(
        tmp_path / "PRS_L2D_STD_20191215092453_20191215092457_0003_0.tif"
    )
    ds = PRISMA(paths=str(tmp_path))
    x, y, t = ds.bounds
    sample = ds[x, y, t]
    assert tuple(sample["image"].shape) == (8, 8, 40)
    assert "bounds" in sample


def test_len(tmp_path):
    _write_prisma_tif(
        tmp_path / "PRS_L2D_STD_20191215092453_20191215092457_0003_0.tif"
    )
    ds = PRISMA(paths=str(tmp_path))
    assert len(ds) == 1


def test_plot(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _write_prisma_tif(
        tmp_path / "PRS_L2D_STD_20191215092453_20191215092457_0003_0.tif"
    )
    ds = PRISMA(paths=str(tmp_path))
    x, y, t = ds.bounds
    sample = ds[x, y, t]
    ds.plot(sample, suptitle="Test")
    plt.close()
