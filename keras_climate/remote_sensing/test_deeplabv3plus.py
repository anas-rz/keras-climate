import numpy as np
import pytest
import keras

from keras_climate.remote_sensing.deeplabv3plus import DeepLabV3Plus, deeplabv3plus_config
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_deeplabv3plus_mapper
from keras_climate.weights.pretrained import deeplabv3plus_resnet50_imagenet_backbone


@pytest.mark.parametrize("variant", ["resnet50", "resnet101"])
def test_builds_and_runs(variant):
    cfg = deeplabv3plus_config(variant)
    model = DeepLabV3Plus(input_shape=(128, 128, 3), num_classes=5, output_stride=16, **cfg)
    x = np.random.randn(1, 128, 128, 3).astype("float32")
    y = model(x)
    assert tuple(y.shape) == (1, 128, 128, 5)


def test_output_stride_8():
    model = DeepLabV3Plus(input_shape=(128, 128, 3), num_classes=5, output_stride=8,
                           backbone_layers=(3, 4, 6, 3))
    x = np.random.randn(1, 128, 128, 3).astype("float32")
    y = model(x)
    assert tuple(y.shape) == (1, 128, 128, 5)


def test_non_multiple_of_32_input_size():
    model = DeepLabV3Plus(input_shape=(130, 130, 3), num_classes=2, backbone_layers=(3, 4, 6, 3))
    x = np.random.randn(1, 130, 130, 3).astype("float32")
    y = model(x)
    assert tuple(y.shape) == (1, 130, 130, 2)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBN(nn.Module):
        def __init__(self, in_ch, out_ch, k, stride=1, dilation=1):
            super().__init__()
            pad = ((k - 1) * dilation) // 2
            self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad,
                                   dilation=dilation, bias=False)
            self.bn = nn.BatchNorm2d(out_ch, eps=1e-5)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_ch, filters, stride=1, dilation=1, downsample=False):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, filters, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(filters, eps=1e-5)
            self.conv2 = nn.Conv2d(filters, filters, 3, stride=stride, padding=dilation,
                                    dilation=dilation, bias=False)
            self.bn2 = nn.BatchNorm2d(filters, eps=1e-5)
            self.conv3 = nn.Conv2d(filters, filters * 4, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(filters * 4, eps=1e-5)
            self.downsample = None
            if downsample:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_ch, filters * 4, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(filters * 4, eps=1e-5),
                )

        def forward(self, x):
            shortcut = self.downsample(x) if self.downsample is not None else x
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            return F.relu(out + shortcut)

    class ResNetBackbone(nn.Module):
        def __init__(self, layer_counts=(3, 4, 6, 3), output_stride=16):
            super().__init__()
            if output_stride == 16:
                strides, dilations = [1, 2, 2, 1], [1, 1, 1, 2]
            else:
                strides, dilations = [1, 2, 1, 1], [1, 1, 2, 4]

            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

            self.in_ch = 64
            self.layer1 = self._make_layer(64, layer_counts[0], strides[0], dilations[0])
            self.layer2 = self._make_layer(128, layer_counts[1], strides[1], dilations[1])
            self.layer3 = self._make_layer(256, layer_counts[2], strides[2], dilations[2])
            self.layer4 = self._make_layer(512, layer_counts[3], strides[3], dilations[3])

        def _make_layer(self, filters, num_blocks, stride, dilation):
            layers_ = [Bottleneck(self.in_ch, filters, stride=stride, dilation=dilation, downsample=True)]
            self.in_ch = filters * 4
            for _ in range(1, num_blocks):
                layers_.append(Bottleneck(self.in_ch, filters, stride=1, dilation=dilation))
            return nn.Sequential(*layers_)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            low_level = x
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return low_level, x

    class ASPP(nn.Module):
        def __init__(self, in_ch, filters=256, rates=(6, 12, 18)):
            super().__init__()
            self.b0 = ConvBN(in_ch, filters, 1)
            self.b6 = ConvBN(in_ch, filters, 3, dilation=rates[0])
            self.b12 = ConvBN(in_ch, filters, 3, dilation=rates[1])
            self.b18 = ConvBN(in_ch, filters, 3, dilation=rates[2])
            self.pool_conv = ConvBN(in_ch, filters, 1)
            self.project = ConvBN(filters * 5, filters, 1)

        def forward(self, x):
            h, w = x.shape[2], x.shape[3]
            feats = [self.b0(x), self.b6(x), self.b12(x), self.b18(x)]
            pooled = x.mean(dim=(2, 3), keepdim=True)
            pooled = self.pool_conv(pooled)
            pooled = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(pooled)
            return self.project(torch.cat(feats, dim=1))

    class TorchDeepLabV3Plus(nn.Module):
        def __init__(self, num_classes=5, layer_counts=(3, 4, 6, 3), output_stride=16,
                     aspp_filters=256, decoder_filters=48):
            super().__init__()
            self.backbone = ResNetBackbone(layer_counts, output_stride)
            self.aspp = ASPP(2048, aspp_filters)
            self.low_level_project = ConvBN(256, decoder_filters, 1)
            self.decoder_conv1 = ConvBN(aspp_filters + decoder_filters, 256, 3)
            self.decoder_conv2 = ConvBN(256, 256, 3)
            self.logits = nn.Conv2d(256, num_classes, 1)

        def forward(self, x):
            input_hw = x.shape[2], x.shape[3]
            low_level, feat = self.backbone(x)
            feat = self.aspp(feat)
            feat = F.interpolate(feat, size=low_level.shape[2:], mode="bilinear", align_corners=False)
            low_level = self.low_level_project(low_level)
            x = torch.cat([feat, low_level], dim=1)
            x = self.decoder_conv1(x)
            x = self.decoder_conv2(x)
            x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
            return self.logits(x)

    torch.manual_seed(0)
    layer_counts = (3, 4, 6, 3)
    torch_model = TorchDeepLabV3Plus(num_classes=5, layer_counts=layer_counts, output_stride=16)
    torch_model.eval()
    with torch.no_grad():
        for m in torch_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_(0.0, 0.1)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)

    state_dict = {}
    for k, v in torch_model.state_dict().items():
        if "num_batches_tracked" in k:
            continue
        k = k[len("backbone."):] if k.startswith("backbone.") else k
        state_dict[k] = v.detach().numpy()

    keras_model = DeepLabV3Plus(input_shape=(128, 128, 3), num_classes=5,
                                 backbone_layers=layer_counts, output_stride=16)
    keras_model(np.zeros((1, 128, 128, 3), dtype="float32"))

    mapper = build_deeplabv3plus_mapper(layer_counts=layer_counts)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    x_np = np.random.randn(1, 3, 128, 128).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x_np)).numpy()

    keras_in = np.transpose(x_np, (0, 2, 3, 1))
    keras_out = keras.ops.convert_to_numpy(keras_model(keras_in, training=False))
    keras_out = np.transpose(keras_out, (0, 3, 1, 2))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"DeepLabV3+ weight port numerical mismatch: max abs diff {max_diff}"


@pytest.mark.pretrained
def test_real_pretrained_imagenet_resnet50_backbone():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    model, report = deeplabv3plus_resnet50_imagenet_backbone(input_shape=(224, 224, 3), num_classes=21)
    assert all(k.startswith(("aspp/", "low_level_project/", "decoder_", "logits/"))
               for k in report["missing_in_source"])
    assert not report["unused_source_keys"]

    x = np.random.rand(1, 224, 224, 3).astype("float32")
    y = keras.ops.convert_to_numpy(model(x, training=False))
    assert y.shape == (1, 224, 224, 21)
    assert np.isfinite(y).all()
