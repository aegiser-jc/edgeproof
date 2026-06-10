from .ma_cross import MACrossStrategy
from .buy_hold import BuyHoldStrategy
from .vol_shock_drift import VolumeShockDriftStrategy

REGISTRY = {
    "ma_cross": MACrossStrategy,
    "buy_hold": BuyHoldStrategy,
    "vol_shock_drift": VolumeShockDriftStrategy,
}

__all__ = ["MACrossStrategy", "BuyHoldStrategy", "VolumeShockDriftStrategy", "REGISTRY"]
