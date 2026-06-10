"""Moving-average crossover — a deliberately simple, well-known baseline.

It is here to exercise the pipeline, not because it is expected to be
profitable. (If a textbook MA cross showed a high Deflated Sharpe net of costs,
that would be a red flag that the harness is leaking, not that you're rich.)
"""
from __future__ import annotations

import pandas as pd

from ..core.strategy import Strategy


class MACrossStrategy(Strategy):
    name = "ma_cross"

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = False):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        long = (fast_ma > slow_ma)
        if self.allow_short:
            sig = long.astype(float) * 2.0 - 1.0   # +1 / -1
        else:
            sig = long.astype(float)               # +1 / 0
        # bars before the slow MA exists -> flat
        sig[slow_ma.isna()] = 0.0
        return sig
