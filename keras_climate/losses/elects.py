"""ELECTS early-classification loss (ported from torchgeo.losses.elects).

Multi-backend port of the loss from `'End-to-end learned early classification
of time series for in-season crop type mapping'
<https://doi.org/10.1016/j.isprsjprs.2022.12.016>`_. Unlike the vast majority
of losses in this repo, ``EarlyRewardLoss`` takes three tensors
(log-probabilities, stopping probabilities, and targets) rather than the
usual ``(y_true, y_pred)`` pair, so it is implemented as a plain callable
class rather than a ``keras.losses.Loss`` subclass, matching the original
torchgeo API.
"""

from keras import ops


def _decision_probability(probability_stopping):
    probability_stopping = ops.convert_to_tensor(probability_stopping)
    remaining = ops.cumprod(1 - probability_stopping[:, :-1], axis=1)
    budget = ops.concatenate(
        [ops.ones_like(probability_stopping[:, :1]), remaining], axis=1
    )
    return ops.concatenate(
        [probability_stopping[:, :-1] * budget[:, :-1], budget[:, -1:]], axis=1
    )


def _nll_loss(log_probs, target, weight=None):
    """Per-element negative log likelihood, `reduction='none'` equivalent."""
    target_idx = ops.cast(ops.expand_dims(target, axis=1), "int32")
    gathered = ops.take_along_axis(log_probs, target_idx, axis=1)
    loss = -ops.squeeze(gathered, axis=1)
    if weight is not None:
        loss = loss * ops.take(weight, ops.cast(target, "int32"), axis=0)
    return loss


class EarlyRewardLoss:
    """ELECTS loss for early classification of time series."""

    def __init__(self, alpha=0.5, epsilon=10, weight=None):
        if not 0 <= alpha <= 1:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0 <= epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")

        self.alpha = alpha
        self.epsilon = epsilon
        self.weight = None if weight is None else ops.convert_to_tensor(weight)

    def __call__(self, log_probs, probability_stopping, target, return_stats=False):
        log_probs = ops.convert_to_tensor(log_probs)
        probability_stopping = ops.convert_to_tensor(probability_stopping)
        target = ops.convert_to_tensor(target)

        if len(log_probs.shape) < 3:
            raise ValueError(
                "log_probs must have shape B x T x C x ..., "
                f"but found {tuple(log_probs.shape)}"
            )

        batch_size, sequence_length, num_classes, *spatial_shape = log_probs.shape
        expected_stopping_shape = (batch_size, sequence_length, *spatial_shape)
        if tuple(probability_stopping.shape) != expected_stopping_shape:
            raise ValueError(
                "probability_stopping must have shape "
                f"{expected_stopping_shape}, but found "
                f"{tuple(probability_stopping.shape)}"
            )

        expected_target_shape = (batch_size, *spatial_shape)
        temporal_target_shape = (batch_size, sequence_length, *spatial_shape)
        if tuple(target.shape) == expected_target_shape:
            target = ops.repeat(
                ops.expand_dims(target, axis=1), sequence_length, axis=1
            )
        elif tuple(target.shape) != temporal_target_shape:
            raise ValueError(
                f"target must have shape {expected_target_shape} or "
                f"{temporal_target_shape}, but found {tuple(target.shape)}"
            )
        target = ops.cast(target, "int32")

        decision_probability = _decision_probability(probability_stopping)
        decision_probability = decision_probability + self.epsilon / sequence_length

        time = ops.arange(sequence_length, dtype=log_probs.dtype)
        time = ops.reshape(time, (1, sequence_length) + (1,) * len(spatial_shape))

        target_idx = ops.expand_dims(target, axis=2)
        gathered = ops.take_along_axis(log_probs, target_idx, axis=2)
        correct_probability = ops.exp(ops.squeeze(gathered, axis=2))

        earliness_reward = ops.mean(
            ops.sum(
                decision_probability
                * correct_probability
                * (1 - time / sequence_length),
                axis=1,
            )
        )

        flat_log_probs = ops.reshape(
            log_probs, (batch_size * sequence_length, num_classes, *spatial_shape)
        )
        flat_target = ops.reshape(
            target, (batch_size * sequence_length, *spatial_shape)
        )
        classification_loss = ops.reshape(
            _nll_loss(flat_log_probs, flat_target, self.weight),
            (batch_size, sequence_length, *spatial_shape),
        )
        classification_loss = ops.mean(
            ops.sum(classification_loss * decision_probability, axis=1)
        )

        loss = self.alpha * classification_loss - (1 - self.alpha) * earliness_reward
        if return_stats:
            stats = {
                "classification_loss": classification_loss,
                "earliness_reward": earliness_reward,
                "probability_making_decision": decision_probability,
            }
            return loss, stats
        return loss
