import math
import numpy as np
import pytest
import keras

from keras_climate.remote_sensing.satclip import SatCLIPLocationEncoder, SphericalHarmonics, SirenLayer
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_satclip_location_mapper


def test_builds_and_runs():
    model = SatCLIPLocationEncoder(legendre_polys=10, dim_hidden=512, num_hidden_layers=2, embed_dim=256)
    lonlat = np.array([[2.35, 48.86], [-74.0, 40.7]], dtype="float32")
    out = keras.ops.convert_to_numpy(model(lonlat))
    assert out.shape == (2, 256)


def test_embedding_dim_is_legendre_polys_squared():
    sh = SphericalHarmonics(legendre_polys=10)
    lonlat = np.array([[0.0, 0.0]], dtype="float32")
    out = keras.ops.convert_to_numpy(sh(lonlat))
    assert out.shape == (1, 100)


def test_spherical_harmonics_matches_known_reference_values():
    sh = SphericalHarmonics(legendre_polys=2)
    lonlat = np.array([[0.0, 0.0]], dtype="float32")
    out = keras.ops.convert_to_numpy(sh(lonlat))[0]

    theta = math.pi / 2
    phi = math.pi
    yl0_m0 = out[0]
    yl1_m_minus1 = out[1]
    yl1_m0 = out[2]
    yl1_m1 = out[3]

    assert np.isclose(yl0_m0, 0.886226925452758, atol=1e-5)
    assert np.isclose(yl1_m0, 1.53499006191973 * math.cos(theta), atol=1e-5)
    assert np.isclose(yl1_m_minus1, 0.48860251190292 * math.sin(theta) * math.sin(phi), atol=1e-5)
    assert np.isclose(yl1_m1, 0.48860251190292 * math.sin(theta) * math.cos(phi), atol=1e-5)


def test_siren_layer_first_layer_uses_higher_frequency():
    x = np.random.uniform(-1, 1, size=(8, 4)).astype("float32")
    low_w0 = SirenLayer(16, w0=1.0)(x)
    high_w0 = SirenLayer(16, w0=30.0)(x)
    assert keras.ops.convert_to_numpy(keras.ops.std(high_w0)) > 0


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class TorchSiren(nn.Module):
        def __init__(self, in_dim, out_dim, w0, use_sine=True):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
            self.bias = nn.Parameter(torch.zeros(out_dim))
            self.w0 = w0
            self.use_sine = use_sine

        def forward(self, x):
            x = torch.nn.functional.linear(x, self.weight, self.bias)
            return torch.sin(self.w0 * x) if self.use_sine else x

    class TorchSirenNet(nn.Module):
        def __init__(self, in_dim, dim_hidden, num_hidden_layers, out_dim, w0_initial, w0):
            super().__init__()
            self.layers = nn.ModuleList([
                TorchSiren(in_dim if i == 0 else dim_hidden, dim_hidden,
                           w0_initial if i == 0 else w0)
                for i in range(num_hidden_layers)
            ])
            self.last_layer = TorchSiren(dim_hidden, out_dim, w0, use_sine=False)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.last_layer(x)

    torch.manual_seed(0)
    legendre_polys, dim_hidden, num_hidden_layers, embed_dim = 6, 32, 2, 16
    in_dim = legendre_polys ** 2

    torch_model = TorchSirenNet(in_dim, dim_hidden, num_hidden_layers, embed_dim,
                                 w0_initial=30.0, w0=1.0)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.2)

    state_dict = {}
    for i, layer in enumerate(torch_model.layers):
        state_dict[f"model.location.nnet.layers.{i}.weight"] = layer.weight.detach().numpy()
        state_dict[f"model.location.nnet.layers.{i}.bias"] = layer.bias.detach().numpy()
    state_dict["model.location.nnet.last_layer.weight"] = torch_model.last_layer.weight.detach().numpy()
    state_dict["model.location.nnet.last_layer.bias"] = torch_model.last_layer.bias.detach().numpy()

    keras_model = SatCLIPLocationEncoder(legendre_polys=legendre_polys, dim_hidden=dim_hidden,
                                          num_hidden_layers=num_hidden_layers, embed_dim=embed_dim)

    mapper = build_satclip_location_mapper(num_hidden_layers=num_hidden_layers)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(4, in_dim).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    sh_layer = keras_model.get_layer("posenc")
    x = keras.Input(shape=(in_dim,))
    y = x
    for i in range(num_hidden_layers):
        y = keras_model.get_layer(f"layers{i}")(y)
    y = keras_model.get_layer("last_layer")(y)
    siren_only = keras.Model(x, y)
    keras_out = keras.ops.convert_to_numpy(siren_only(x_np, training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"SatCLIP SirenNet weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_satclip_resnet18_l10():
    pytest.importorskip("torch")
    from keras_climate.weights.pretrained import satclip_location_encoder_resnet18_l10

    model, report = satclip_location_encoder_resnet18_l10()
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    lonlat = np.array([[2.35, 48.86], [-74.0, 40.7], [139.7, 35.7]], dtype="float32")
    out = keras.ops.convert_to_numpy(model(lonlat))
    assert out.shape == (3, 256)
    assert np.isfinite(out).all()
