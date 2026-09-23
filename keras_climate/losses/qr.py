"""QR / RQ losses for learning on the prior (ported from torchgeo.losses.qr).

From `'Resolving label uncertainty with implicit posterior models'
<https://arxiv.org/abs/2202.14000>`_. Implemented as plain callable classes
(``loss(probs, target)``) rather than ``keras.losses.Loss`` subclasses to
keep the original torchgeo call signature.
"""

from keras import ops

_NORMALIZE_EPS = 1e-12


class QRLoss:
    """The QR (forward) loss between class probabilities and predictions."""

    def __init__(self, eps=1e-8):
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        self.eps = eps

    def __call__(self, probs, target):
        q = ops.convert_to_tensor(probs)
        target = ops.convert_to_tensor(target)

        q_bar = ops.mean(q, axis=(0, 2, 3))
        qbar_log_s = ops.sum(q_bar * ops.log(q_bar + self.eps))

        q_log_p = ops.mean(ops.sum(q * ops.log(target + self.eps), axis=1))

        return qbar_log_s - q_log_p


class RQLoss:
    """The RQ (backwards) loss between class probabilities and predictions."""

    def __init__(self, eps=1e-8):
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        self.eps = eps

    def __call__(self, probs, target):
        q = ops.convert_to_tensor(probs)
        target = ops.convert_to_tensor(target)

        # Manually normalize due to https://github.com/pytorch/pytorch/issues/70100
        l1_norm = ops.sum(ops.abs(q), axis=(0, 2, 3), keepdims=True)
        z = q / ops.maximum(l1_norm, self.eps)

        zt = z * target
        zt_l1_norm = ops.sum(ops.abs(zt), axis=1, keepdims=True)
        r = zt / ops.maximum(zt_l1_norm, _NORMALIZE_EPS)

        loss = ops.mean(
            ops.sum(r * (ops.log(r + self.eps) - ops.log(q + self.eps)), axis=1)
        )

        return loss
