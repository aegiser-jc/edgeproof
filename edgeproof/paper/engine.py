"""Paper-trading engine — runs a strategy live against real prices, fake money.

It acts only on **closed** bars (the still-forming current bar is dropped), so
it has the same no-look-ahead discipline as the backtester. Each tick is
idempotent: if no new bar has closed since last time, nothing happens. State is
persisted to JSON after every tick, so the loop survives restarts and can be
driven by a long-running process or a cron/scheduler.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..data import fetch_klines, INTERVAL_MS
from ..strategies import REGISTRY
from .portfolio import PaperPortfolio

_STATE_DIR = Path(__file__).resolve().parents[2] / "paper_state"
_LOOKBACK_BARS = 500  # enough history to warm up MA(200) / vol_window(96)


def _now_ms() -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)


class PaperTrader:
    def __init__(self, symbol, interval, strategy, params, portfolio: PaperPortfolio,
                 *, name=None, last_bar_time=None, started_at=None):
        self.symbol = symbol
        self.interval = interval
        self.strategy = strategy
        self.params = params
        self.portfolio = portfolio
        self.name = name or f"{symbol}_{interval}_{strategy}"
        self.last_bar_time = last_bar_time
        self.started_at = started_at or pd.Timestamp.now(tz="UTC").isoformat()

    # --- persistence ---------------------------------------------------------
    @staticmethod
    def _path(name: str) -> Path:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        return _STATE_DIR / f"{name}.json"

    @classmethod
    def exists(cls, name: str) -> bool:
        return cls._path(name).exists()

    @classmethod
    def load(cls, name: str) -> "PaperTrader":
        d = json.loads(cls._path(name).read_text())
        return cls(
            symbol=d["symbol"], interval=d["interval"], strategy=d["strategy"],
            params=d["params"], portfolio=PaperPortfolio.from_dict(d["portfolio"]),
            name=name, last_bar_time=d.get("last_bar_time"),
            started_at=d.get("started_at"),
        )

    def save(self) -> None:
        d = {
            "symbol": self.symbol, "interval": self.interval,
            "strategy": self.strategy, "params": self.params,
            "started_at": self.started_at, "last_bar_time": self.last_bar_time,
            "portfolio": self.portfolio.to_dict(),
        }
        self._path(self.name).write_text(json.dumps(d, indent=2, default=str))

    # --- core ----------------------------------------------------------------
    def _latest_closed(self) -> pd.DataFrame:
        data = fetch_klines(self.symbol, self.interval, _LOOKBACK_BARS, force_refresh=True)
        bar_ms = INTERVAL_MS[self.interval]
        now = _now_ms()
        # a bar is closed once its open_time + interval <= now
        open_ms = data.index.asi8 // 1_000_000  # bar open time in ms (UTC)
        closed = data[(open_ms + bar_ms) <= now]
        return closed

    def tick(self) -> dict:
        """One iteration. Acts only if a new bar has closed. Returns a status dict."""
        data = self._latest_closed()
        if data.empty:
            return {"acted": False, "reason": "no closed bars"}

        last_time = data.index[-1]
        last_iso = last_time.isoformat()
        if self.last_bar_time is not None and last_iso <= self.last_bar_time:
            return {"acted": False, "reason": "no new bar", "last_bar": self.last_bar_time}

        strat = REGISTRY[self.strategy](**self.params)
        target = float(strat.generate_signals(data).iloc[-1])
        price = float(data["close"].iloc[-1])

        trade = self.portfolio.rebalance_to(target, price, last_iso)
        self.portfolio.snapshot(last_iso, price)
        self.last_bar_time = last_iso
        self.save()

        return {
            "acted": True, "bar": last_iso, "price": price,
            "target_fraction": target, "traded": trade is not None,
            "trade": trade, "equity": self.portfolio.equity(price),
        }

    def run(self, poll_seconds: int = 60, max_iter: int | None = None) -> None:
        i = 0
        while max_iter is None or i < max_iter:
            status = self.tick()
            yield status
            i += 1
            if max_iter is not None and i >= max_iter:
                break
            time.sleep(poll_seconds)
