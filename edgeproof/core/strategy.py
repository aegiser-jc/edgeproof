"""Strategy interface.

A strategy maps market data to a *target position* series. The contract that
keeps the harness honest:

    generate_signals(data) -> pd.Series aligned to data.index, values in [-1, 1]

  * The value at bar t is the position you WANT after observing bar t's close.
  * +1 = fully long, 0 = flat, -1 = fully short (fractions allowed).
  * The strategy may only use information up to and including bar t. It must
    NOT read data[t+1:]. The backtester additionally executes the position at
    bar t+1, so even a subtly leaky signal cannot trade on the same bar it was
    computed from.
"""
from __future__ import annotations

import pandas as pd


class Strategy:
    name = "base"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({params})"
