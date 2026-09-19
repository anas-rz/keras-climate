import numpy as np
import pytest
import keras

from keras_climate.remote_sensing.ssl4eo import SSL4EOResNet50
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_ssl4eo_mapper


def test_builds_and_runs():
    model = SSL4EOResNet50(input_shape=(64, 64, 13))
    x = np.random.randn(2, 64, 64, 13).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (2, 2, 2, 2048)


def test_thirteen_band_stem():
    model = SSL4EOResNet50(input_shape=(32, 32, 13))
    stem_conv = model.get_layer("backbone_stem").conv
    assert stem_conv.kernel.shape[2] == 13


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torchvision

    torch.manual_seed(0)
    torch_model = torchvision.models.resnet50(weights=None)
    torch_model.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
    torch_model.eval()
    with torch.no_grad():
        for m in torch_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_(0.0, 0.1)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
            elif isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.normal_(0.0, 0.05)

    state_dict = {}
    for k, v in torch_model.state_dict().items():
        if "num_batches_tracked" in k or k.startswith("fc."):
            continue
        state_dict[k] = v.detach().numpy()

    keras_model = SSL4EOResNet50(input_shape=(64, 64, 13))
    keras_model(np.zeros((1, 64, 64, 13), dtype="float32"))

    mapper = build_ssl4eo_mapper(layer_counts=(3, 4, 6, 3))
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(1, 13, 64, 64).astype("float32")
    with torch.no_grad():
        x = torch_model.conv1(torch.from_numpy(x_np))
        x = torch_model.bn1(x)
        x = torch_model.relu(x)
        x = torch_model.maxpool(x)
        x = torch_model.layer1(x)
        x = torch_model.layer2(x)
        x = torch_model.layer3(x)
        torch_out = torch_model.layer4(x).numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    keras_out = keras.ops.convert_to_numpy(keras_model(keras_in, training=False))
    keras_out = np.transpose(keras_out, (0, 3, 1, 2))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"SSL4EO weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_ssl4eo_resnet50_moco():
    pytest.importorskip("torch")
    from keras_climate.weights.pretrained import ssl4eo_resnet50_moco

    model, report = ssl4eo_resnet50_moco(input_shape=(224, 224, 13))
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x = np.random.rand(1, 224, 224, 13).astype("float32")
    y = keras.ops.convert_to_numpy(model(x))
    assert y.shape == (1, 7, 7, 2048)
    assert np.isfinite(y).all()
