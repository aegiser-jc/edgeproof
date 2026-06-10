"""Volume-shock drift — a theory-backed proxy for post-news drift.

Motivation (from ../research/findings-trading-thesis.md): the only *tradable*
residual of news/sentiment that survives in the literature is the **post-news
drift**, concentrated in small/illiquid names and in negative news — i.e. the
slow continuation AFTER the initial (untradeable) jump.

We don't have a news feed yet, but information arrival leaves a fingerprint in
the tape: an abnormally large price move on abnormally high volume. This
strategy detects that fingerprint and bets the move *continues* (drift) for a
fixed holding period.

This is the honest test of the user's thesis: if there is no net-of-cost drift
even on the clearest information-arrival events, the thesis is dead; if there
is, it should be stronger on smaller/less liquid pairs (test that by running the
same params on a large vs a small symbol).

Causal by construction: the event at bar t uses only volume/return through t;
rolling baselines use only past bars; the backtester then enters at t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.strategy import Strategy


class VolumeShockDriftStrategy(Strategy):
    name = "vol_shock_drift"

    def __init__(
        self,
        vol_window: int = 96,      # baseline window for "normal" volume
        vol_z: float = 3.0,        # how abnormal the volume must be (z-score)
        ret_thresh: float = 0.01,  # minimum |bar return| to count as an event
        hold_bars: int = 6,        # how long to ride the drift
        mode: str = "momentum",    # 'momentum' (continuation) or 'reversal'
        event_side: str = "both",  # 'both' | 'up' | 'down' (e.g. down = negative-news only)
        long_only: bool = False,   # clamp shorts to flat (spot-only venues)
    ):
        if hold_bars < 1:
            raise ValueError("hold_bars must be >= 1")
        if mode not in ("momentum", "reversal"):
            raise ValueError("mode must be 'momentum' or 'reversal'")
        if event_side not in ("both", "up", "down"):
            raise ValueError("event_side must be 'both', 'up' or 'down'")
        self.vol_window = vol_window
        self.vol_z = vol_z
        self.ret_thresh = ret_thresh
        self.hold_bars = hold_bars
        self.mode = mode
        self.event_side = event_side
        self.long_only = long_only

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        vol = data["volume"]
        ret = close.pct_change()

        vol_mean = vol.rolling(self.vol_window).mean()
        vol_std = vol.rolling(self.vol_window).std()
        vol_z = (vol - vol_mean) / vol_std.replace(0.0, np.nan)

        is_event = (vol_z >= self.vol_z) & (ret.abs() >= self.ret_thresh)
        if self.event_side == "up":
            is_event &= ret > 0
        elif self.event_side == "down":
            is_event &= ret < 0

        direction = np.sign(ret)
        if self.mode == "reversal":
            direction = -direction

        raw = pd.Series(np.where(is_event, direction, np.nan), index=close.index)
        # hold the position for `hold_bars` bars; a fresh event overrides
        position = raw.ffill(limit=self.hold_bars - 1).fillna(0.0)

        # warmup: no signal until the volume baseline exists
        position[vol_std.isna()] = 0.0
        if self.long_only:
            position = position.clip(lower=0.0)
        return position.clip(-1.0, 1.0)
