import numpy as np
import pytest
import keras

from keras_climate.operators.deeponet import DeepONet
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_deeponet_mapper


def test_builds_and_runs():
    model = DeepONet(num_sensors=50, coord_dim=2, branch_units=(32, 16), trunk_units=(32, 16))
    branch = np.random.randn(4, 50).astype("float32")
    trunk = np.random.randn(4, 2).astype("float32")
    y = keras.ops.convert_to_numpy(model([branch, trunk]))
    assert y.shape == (4, 1)


def test_mismatched_final_widths_raises():
    with pytest.raises(AssertionError):
        DeepONet(num_sensors=10, coord_dim=1, branch_units=(8, 16), trunk_units=(8, 8))


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchDeepONet(nn.Module):
        def __init__(self, num_sensors, coord_dim, branch_units, trunk_units):
            super().__init__()
            self.branch_units = branch_units
            self.trunk_units = trunk_units
            dims = [num_sensors] + list(branch_units)
            for i in range(len(branch_units)):
                setattr(self, f"branch_fc{i}", nn.Linear(dims[i], dims[i + 1]))
            dims = [coord_dim] + list(trunk_units)
            for i in range(len(trunk_units)):
                setattr(self, f"trunk_fc{i}", nn.Linear(dims[i], dims[i + 1]))
            self.bias = nn.Parameter(torch.zeros(()))

        def forward(self, branch_in, trunk_in):
            b = branch_in
            for i in range(len(self.branch_units)):
                b = getattr(self, f"branch_fc{i}")(b)
                if i < len(self.branch_units) - 1:
                    b = F.relu(b)
            t = trunk_in
            for i in range(len(self.trunk_units)):
                t = F.relu(getattr(self, f"trunk_fc{i}")(t))
            out = (b * t).sum(dim=-1, keepdim=True)
            return out + self.bias

    torch.manual_seed(0)
    num_sensors, coord_dim = 20, 2
    branch_units, trunk_units = (16, 8), (16, 8)

    torch_model = TorchDeepONet(num_sensors, coord_dim, branch_units, trunk_units)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.3)

    state_dict = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}

    keras_model = DeepONet(num_sensors=num_sensors, coord_dim=coord_dim, branch_units=branch_units,
                            trunk_units=trunk_units)

    mapper = build_deeponet_mapper(num_branch_layers=len(branch_units), num_trunk_layers=len(trunk_units))
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    branch_np = np.random.randn(4, num_sensors).astype("float32")
    trunk_np = np.random.randn(4, coord_dim).astype("float32")
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(branch_np), torch.from_numpy(trunk_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model([branch_np, trunk_np], training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"DeepONet weight port numerical mismatch: max abs diff {max_diff}"
