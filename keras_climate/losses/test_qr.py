import numpy as np
import pytest
from keras import ops

from keras_climate.losses import QRLoss, RQLoss


def test_loss_on_prior_simple():
    probs = np.random.rand(2, 4, 10, 10).astype("float32")
    log_probs = np.log(probs)
    targets = np.random.rand(2, 4, 10, 10).astype("float32")
    loss = QRLoss()(log_probs, targets)
    assert loss.shape == ()


def test_loss_on_prior_reversed_kl_simple():
    probs = np.random.rand(2, 4, 10, 10).astype("float32")
    log_probs = np.log(probs)
    targets = np.random.rand(2, 4, 10, 10).astype("float32")
    loss = RQLoss()(log_probs, targets)
    assert loss.shape == ()


def test_invalid_eps_value():
    with pytest.raises(ValueError, match="Invalid epsilon value"):
        QRLoss(eps=-1.0)

    with pytest.raises(ValueError, match="Invalid epsilon value"):
        RQLoss(eps=-1.0)


def test_eps_prevents_nan():
    logits = np.random.rand(2, 4, 10, 10).astype("float32")
    probs = ops.convert_to_numpy(ops.softmax(logits, axis=1))
    labels = np.random.randint(0, 4, (2, 10, 10))
    targets = np.eye(4, dtype="float32")[labels].transpose(0, 3, 1, 2)

    loss = QRLoss(eps=1e-8)(probs, targets)
    assert np.isfinite(ops.convert_to_numpy(loss))

    loss = RQLoss(eps=1e-8)(probs, targets)
    assert np.isfinite(ops.convert_to_numpy(loss))


def test_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    probs = np.random.rand(2, 4, 5, 5).astype("float32")
    targets = np.random.rand(2, 4, 5, 5).astype("float32")
    eps = 1e-8

    q = torch.tensor(probs)
    t = torch.tensor(targets)

    q_bar = q.mean(dim=(0, 2, 3))
    qbar_log_s = (q_bar * torch.log(q_bar + eps)).sum()
    q_log_p = torch.einsum("bcxy,bcxy->bxy", q, torch.log(t + eps)).mean()
    expected_qr = qbar_log_s - q_log_p

    z = q / q.norm(p=1, dim=(0, 2, 3), keepdim=True).clamp_min(eps).expand_as(q)
    r = F.normalize(z * t, p=1, dim=1)
    expected_rq = torch.einsum(
        "bcxy,bcxy->bxy", r, torch.log(r + eps) - torch.log(q + eps)
    ).mean()

    actual_qr = QRLoss(eps=eps)(probs, targets)
    actual_rq = RQLoss(eps=eps)(probs, targets)

    np.testing.assert_allclose(
        ops.convert_to_numpy(actual_qr), expected_qr.numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(actual_rq), expected_rq.numpy(), atol=1e-5
    )
