"""Event-driven backtester that cannot look ahead and always charges costs.

No-look-ahead, by construction
------------------------------
The strategy's target position for bar t (`target[t]`) is the position we want
after seeing bar t's close. We can only act on it at the *next* bar's open, so
the position actually *held during* bar t is ``target[t-1]``. We implement that
as a one-bar shift. A strategy therefore physically cannot earn the return of
the same bar that produced its signal.

Costs, always
-------------
Whenever the held position changes (a trade), we charge
``turnover * cost_rate`` where ``cost_rate`` bundles exchange fee, half the
bid/ask spread, and a slippage estimate (all in basis points). Turnover for a
flat->long entry is 1.0; a full long->short flip is 2.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """All values in basis points (1 bp = 0.01%). Charged per unit of turnover.

    Defaults are deliberately *pessimistic* for a small retail account on a
    liquid crypto pair — the whole point of EdgeProof is that an edge must
    survive realistic frictions, not vanish the moment you trade it.
    """
    fee_bps: float = 7.5        # taker fee per side (e.g. ~0.075% spot)
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0

    @property
    def rate(self) -> float:
        return (self.fee_bps + self.half_spread_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    equity: pd.Series          # net equity curve, starts at 1.0
    net_returns: pd.Series     # per-bar net returns
    gross_returns: pd.Series   # per-bar returns before costs
    position: pd.Series        # position actually held during each bar
    turnover: pd.Series        # |change in held position| per bar
    costs: pd.Series           # cost drag per bar (fraction of equity)
    benchmark_equity: pd.Series  # buy & hold, same cost on entry
    interval: str
    cost_model: CostModel
    capacity_warnings: list

    @property
    def n_trades(self) -> int:
        return int((self.turnover > 1e-9).sum())


def run_backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    *,
    interval: str = "1h",
    cost_model: CostModel | None = None,
    capacity_usd: float | None = None,
    capacity_volume_frac: float = 0.01,
) -> BacktestResult:
    """Run a vectorised but strictly causal backtest.

    Parameters
    ----------
    data : OHLCV frame (needs at least a 'close' column; 'quote_volume' enables
           capacity checks).
    signals : target position per bar, values in [-1, 1].
    capacity_usd : if given, the notional you intend to trade. Bars where this
           exceeds `capacity_volume_frac` of the bar's quote volume are flagged
           — that's where your "edge" may not be tradable at size.
    """
    cost_model = cost_model or CostModel()
    close = data["close"].astype(float)

    target = signals.reindex(close.index).fillna(0.0).clip(-1.0, 1.0)
    # Position held during bar t was decided at the close of bar t-1.
    held = target.shift(1).fillna(0.0)

    bar_ret = close.pct_change().fillna(0.0)
    gross = held * bar_ret

    # Turnover: the trade that establishes `held[t]` happens at the open of t.
    turnover = held.diff().abs()
    turnover.iloc[0] = abs(held.iloc[0])
    costs = turnover * cost_model.rate
    net = gross - costs

    equity = (1.0 + net).cumprod()

    # Buy & hold benchmark (one entry cost, then just hold).
    bh_cost = pd.Series(0.0, index=close.index)
    if len(bh_cost) > 1:
        bh_cost.iloc[1] = cost_model.rate  # enter at the second bar
    bench = (1.0 + bar_ret - bh_cost).cumprod()

    capacity_warnings: list = []
    if capacity_usd is not None and "quote_volume" in data.columns:
        qv = data["quote_volume"].astype(float)
        traded = turnover * capacity_usd
        flagged = traded > (capacity_volume_frac * qv)
        n_flag = int(flagged.sum())
        if n_flag:
            capacity_warnings.append(
                f"{n_flag} of {self_trade_count(turnover)} trades exceed "
                f"{capacity_volume_frac:.1%} of bar volume at ${capacity_usd:,.0f} "
                f"notional — slippage will likely be worse than modelled."
            )

    return BacktestResult(
        equity=equity,
        net_returns=net,
        gross_returns=gross,
        position=held,
        turnover=turnover,
        costs=costs,
        benchmark_equity=bench,
        interval=interval,
        cost_model=cost_model,
        capacity_warnings=capacity_warnings,
    )


def self_trade_count(turnover: pd.Series) -> int:
    return int((turnover > 1e-9).sum())
