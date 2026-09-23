import numpy as np
import pytest
import keras
from keras import ops

from keras_climate.losses import EarlyRewardLoss
from keras_climate.losses.elects import _decision_probability


def _log_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


@pytest.fixture
def inputs():
    logits = np.array(
        [
            [[2.0, 1.0], [1.0, 2.0], [2.0, 1.0]],
            [[1.0, 2.0], [2.0, 1.0], [1.0, 2.0]],
        ],
        dtype="float32",
    )
    log_probs = _log_softmax(logits, axis=-1)
    probability_stopping = np.array(
        [[0.2, 0.5, 0.9], [0.4, 0.5, 0.9]], dtype="float32"
    )
    target = np.array([0, 1])
    return log_probs, probability_stopping, target


def test_decision_probability():
    probability_stopping = np.array([[0.2, 0.5, 0.9]], dtype="float32")
    actual = ops.convert_to_numpy(_decision_probability(probability_stopping))
    expected = np.array([[0.2, 0.4, 0.4]], dtype="float32")
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_single_time_step():
    actual = ops.convert_to_numpy(
        _decision_probability(np.array([[0.5], [0.1]], dtype="float32"))
    )
    np.testing.assert_array_equal(actual, np.ones((2, 1), dtype="float32"))


def test_forward(inputs):
    log_probs, probability_stopping, target = inputs
    loss = EarlyRewardLoss(epsilon=0)(log_probs, probability_stopping, target)
    assert loss.shape == ()
    assert np.isfinite(ops.convert_to_numpy(loss))


def test_return_stats(inputs):
    log_probs, probability_stopping, target = inputs
    output = EarlyRewardLoss()(log_probs, probability_stopping, target, True)
    assert isinstance(output, tuple)
    loss, stats = output
    assert loss.shape == ()
    assert set(stats) == {
        "classification_loss",
        "earliness_reward",
        "probability_making_decision",
    }
    assert tuple(stats["probability_making_decision"].shape) == tuple(
        probability_stopping.shape
    )


def test_temporal_target(inputs):
    log_probs, probability_stopping, target = inputs
    temporal_target = np.repeat(
        target[:, None], log_probs.shape[1], axis=1
    )
    expected = EarlyRewardLoss()(log_probs, probability_stopping, target)
    actual = EarlyRewardLoss()(log_probs, probability_stopping, temporal_target)
    np.testing.assert_allclose(
        ops.convert_to_numpy(actual), ops.convert_to_numpy(expected), atol=1e-6
    )


def test_spatial_target():
    log_probs = _log_softmax(
        np.random.randn(2, 3, 4, 5, 6).astype("float32"), axis=2
    )
    probability_stopping = np.random.rand(2, 3, 5, 6).astype("float32")
    target = np.random.randint(0, 4, (2, 5, 6))
    loss = EarlyRewardLoss()(log_probs, probability_stopping, target)
    assert loss.shape == ()


def test_weight(inputs):
    log_probs, probability_stopping, target = inputs
    loss = EarlyRewardLoss(weight=np.array([1.0, 2.0], dtype="float32"))(
        log_probs, probability_stopping, target
    )
    assert np.isfinite(ops.convert_to_numpy(loss))


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="Invalid alpha value"):
        EarlyRewardLoss(alpha=alpha)


def test_invalid_epsilon():
    with pytest.raises(ValueError, match="Invalid epsilon value"):
        EarlyRewardLoss(epsilon=-1)


def test_invalid_log_probs_shape():
    with pytest.raises(ValueError, match="log_probs must have shape"):
        EarlyRewardLoss()(
            np.random.rand(2, 3).astype("float32"),
            np.random.rand(2, 3).astype("float32"),
            np.ones(2),
        )


def test_invalid_stopping_shape():
    with pytest.raises(ValueError, match="probability_stopping must have shape"):
        EarlyRewardLoss()(
            np.random.rand(2, 3, 4).astype("float32"),
            np.random.rand(2, 4).astype("float32"),
            np.ones(2, dtype="int64"),
        )


def test_invalid_target_shape():
    with pytest.raises(ValueError, match="target must have shape"):
        EarlyRewardLoss()(
            np.random.rand(2, 3, 4).astype("float32"),
            np.random.rand(2, 3).astype("float32"),
            np.ones((2, 2), dtype="int64"),
        )


def test_matches_pytorch_reference(inputs):
    torch = pytest.importorskip("torch")
    log_probs, probability_stopping, target = inputs

    t_log_probs = torch.tensor(log_probs)
    t_stopping = torch.tensor(probability_stopping)
    t_target = torch.tensor(target, dtype=torch.long)

    remaining = torch.cumprod(1 - t_stopping[:, :-1], dim=1)
    budget = torch.cat([torch.ones_like(t_stopping[:, :1]), remaining], dim=1)
    decision_probability = torch.cat(
        [t_stopping[:, :-1] * budget[:, :-1], budget[:, -1:]], dim=1
    )
    decision_probability = decision_probability + 0 / t_log_probs.shape[1]

    time = torch.arange(t_log_probs.shape[1], dtype=t_log_probs.dtype)
    correct_probability = (
        t_log_probs.gather(dim=2, index=t_target[:, None, None].expand(-1, 3, 1))
        .squeeze(2)
        .exp()
    )
    earliness_reward = (
        (
            decision_probability
            * correct_probability
            * (1 - time / t_log_probs.shape[1])
        )
        .sum(dim=1)
        .mean()
    )
    nll = torch.nn.functional.nll_loss(
        t_log_probs.reshape(-1, 2), t_target.repeat_interleave(3), reduction="none"
    ).reshape(2, 3)
    classification_loss = (nll * decision_probability).sum(dim=1).mean()
    expected = 0.5 * classification_loss - 0.5 * earliness_reward

    actual = EarlyRewardLoss(epsilon=0)(log_probs, probability_stopping, target)
    np.testing.assert_allclose(
        ops.convert_to_numpy(actual), expected.detach().numpy(), atol=1e-5
    )
