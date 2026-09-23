"""Multi-backend Keras port of torchgeo.losses."""

from .elects import EarlyRewardLoss
from .qr import QRLoss, RQLoss

__all__ = ("EarlyRewardLoss", "QRLoss", "RQLoss")
