"""
keras_climate.remote_sensing.satclip
----------------------------------------
SatCLIP (Klemmer et al. 2023, Microsoft): a CLIP-style dual encoder,
contrastively trained so a (lon, lat) coordinate's embedding predicts the
embedding of co-located Sentinel-2 imagery. Only the **location encoder**
is implemented here: a real spherical-harmonics positional encoding of the
coordinate, followed by a SirenNet (sinusoidal-activation MLP). It has no
image input, is tiny (~1MB of the checkpoint's 57MB), is architecturally
unlike anything else in this repo's roster, and is directly useful
standalone for geospatial ML - unlike the image side, which duplicates
backbones already covered by SatMAE/Prithvi/CROMA/Clay/AnySat/SSL4EO.

The spherical harmonics use the official checkpoint's default
`harmonics_calculation="analytic"` sign/normalization convention (a
~40,000-line sympy-generated closed-form expansion upstream, one function
per (l, m) pair - not itself reproduced here, but reverse-engineered from
its output values): standard 4π-normalized real spherical harmonics
without the Condon-Shortley phase, *except* the m=0 (zonal) terms carry
an extra factor of π that the official "analytic" functions apply but the
"closed-form" alternative in the same codebase does not - confirmed
against known reference outputs (`Yl0_m0 = 0.886226925...`, `Yl1_m0 =
1.534990062...`, both exactly `pi` times the standard-normalized value;
`Yl1_{-1,1} = 0.488602512...`, matching the standard normalization
directly for m != 0).
"""

import math
import keras
from keras import layers, ops


class SphericalHarmonics(layers.Layer):
    """Real (4π-normalized) spherical harmonics of a (lon, lat) coordinate
    in degrees, computed for degree l in [0, L) and order m in [-l, l],
    stacked into an L^2-dim vector in (l, m) ascending order - matching
    the official `SphericalHarmonics.forward`'s loop order and input
    transform (`phi = deg2rad(lon + 180)`, `theta = deg2rad(lat + 90)`).
    No learnable weights (see module docstring for the normalization
    caveat)."""

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
        """Associated Legendre polynomial P_l^m(x), Condon-Shortley-phase
        free (matches the official "analytic" sign convention, unlike the
        phase-included "closed-form" alternative also present upstream)."""
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
        return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / (4.0 * math.pi * math.factorial(l + m)))

    def _sh(self, m, l, phi, x):
        if m == 0:
            # The official "analytic" convention scales zonal (m=0) terms
            # by an extra factor of pi relative to the standard
            # normalization - see module docstring.
            return math.pi * self._norm(l, 0) * self._legendre(l, 0, x)
        if m > 0:
            return math.sqrt(2.0) * self._norm(l, m) * ops.cos(m * phi) * self._legendre(l, m, x)
        return math.sqrt(2.0) * self._norm(l, -m) * ops.sin(-m * phi) * self._legendre(l, -m, x)


class SirenLayer(layers.Layer):
    """One `sin(w0 * (Wx + b))` layer (Sitzmann et al.'s SIREN), or a plain
    linear layer when `use_sine=False` (the official SirenNet's final
    `last_layer`). Stores `kernel`/`bias` directly (not via a nested Dense
    sublayer) so the Keras weight path matches the official checkpoint's
    flat `layers.{i}.weight/bias` / `last_layer.weight/bias` naming
    one-to-one."""

    def __init__(self, units, w0=1.0, use_sine=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.w0 = w0
        self.use_sine = use_sine

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(in_dim, self.units), initializer="glorot_uniform", name="kernel")
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias")
        super().build(input_shape)

    def call(self, x):
        x = ops.matmul(x, self.kernel) + self.bias
        return ops.sin(self.w0 * x) if self.use_sine else x


def SatCLIPLocationEncoder(legendre_polys=10, dim_hidden=512, num_hidden_layers=2,
                            embed_dim=256, w0_initial=30.0, w0=1.0,
                            name="satclip_location_encoder"):
    """Inputs: `lonlat` of shape (B, 2) - (longitude, latitude) in degrees.
    Output: `(B, embed_dim)` location embeddings."""
    lonlat_in = keras.Input(shape=(2,), name="lonlat")
    x = SphericalHarmonics(legendre_polys, name="posenc")(lonlat_in)

    for i in range(num_hidden_layers):
        x = SirenLayer(dim_hidden, w0=(w0_initial if i == 0 else w0), use_sine=True,
                        name=f"layers{i}")(x)
    out = SirenLayer(embed_dim, w0=w0, use_sine=False, name="last_layer")(x)

    return keras.Model(lonlat_in, out, name=name)
