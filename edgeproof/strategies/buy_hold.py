"""Always fully long — the honest benchmark every strategy must beat *net*."""
from __future__ import annotations

import pandas as pd

from ..core.strategy import Strategy


class BuyHoldStrategy(Strategy):
    name = "buy_hold"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index)
