import math
import keras
from keras import layers, ops


class SphericalHarmonics(layers.Layer):

    def __init__(self, legendre_polys=10, **kwargs):
        super().__init__(**kwargs)
        self.L = legendre_polys
        self.embedding_dim = self.L * self.L

    def call(self, lonlat):
        lon, lat = lonlat[:, 0], lonlat[:, 1]
        phi = (lon + 180.0) * (math.pi / 180.0)
        theta = (lat + 90.0) * (math.pi / 180.0)
        x = ops.cos(theta)

        columns = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                columns.append(self._sh(m, l, phi, x))
        return ops.stack(columns, axis=-1)

    @staticmethod
    def _legendre(l, m, x):
        pmm = ops.ones_like(x)
        if m > 0:
            somx2 = ops.sqrt((1.0 - x) * (1.0 + x))
            fact = 1.0
            for _ in range(m):
                pmm = pmm * fact * somx2
                fact += 2.0
        if l == m:
            return pmm
        pmmp1 = x * (2.0 * m + 1.0) * pmm
        if l == m + 1:
            return pmmp1
        pll = pmmp1
        for ll in range(m + 2, l + 1):
            pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
            pmm, pmmp1 = pmmp1, pll
        return pll

    @staticmethod
    def _norm(l, m):
        return math.sqrt(
            (2.0 * l + 1.0)
            * math.factorial(l - m)
            / (4.0 * math.pi * math.factorial(l + m))
        )

    def _sh(self, m, l, phi, x):
        if m == 0:
            return math.pi * self._norm(l, 0) * self._legendre(l, 0, x)
        if m > 0:
            return (
                math.sqrt(2.0)
                * self._norm(l, m)
                * ops.cos(m * phi)
                * self._legendre(l, m, x)
            )
        return (
            math.sqrt(2.0)
            * self._norm(l, -m)
            * ops.sin(-m * phi)
            * self._legendre(l, -m, x)
        )


class SirenLayer(layers.Layer):

    def __init__(self, units, w0=1.0, use_sine=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.w0 = w0
        self.use_sine = use_sine

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(in_dim, self.units), initializer="glorot_uniform", name="kernel"
        )
        self.bias = self.add_weight(
            shape=(self.units,), initializer="zeros", name="bias"
        )
        super().build(input_shape)

    def call(self, x):
        x = ops.matmul(x, self.kernel) + self.bias
        return ops.sin(self.w0 * x) if self.use_sine else x


def SatCLIPLocationEncoder(
    legendre_polys=10,
    dim_hidden=512,
    num_hidden_layers=2,
    embed_dim=256,
    w0_initial=30.0,
    w0=1.0,
    name="satclip_location_encoder",
):
    lonlat_in = keras.Input(shape=(2,), name="lonlat")
    x = SphericalHarmonics(legendre_polys, name="posenc")(lonlat_in)

    for i in range(num_hidden_layers):
        x = SirenLayer(
            dim_hidden,
            w0=(w0_initial if i == 0 else w0),
            use_sine=True,
            name=f"layers{i}",
        )(x)
    out = SirenLayer(embed_dim, w0=w0, use_sine=False, name="last_layer")(x)

    return keras.Model(lonlat_in, out, name=name)
