"""Fake-money portfolio that rebalances to a target position fraction.

Same economics as the backtester so paper results are comparable to backtests:
a target of f in [-1, 1] means hold notional ``f * equity`` in the asset. Every
rebalance charges the same fee+spread+slippage rate as the backtest CostModel.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# don't bother trading for rebalances smaller than this fraction of equity
_MIN_TRADE_FRAC = 1e-3


@dataclass
class PaperPortfolio:
    cost_rate: float                 # fraction per unit traded notional
    cash: float = 10_000.0
    position_units: float = 0.0      # asset units held (can be negative = short)
    initial_equity: float = 10_000.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    def equity(self, price: float) -> float:
        return self.cash + self.position_units * price

    def position_fraction(self, price: float) -> float:
        eq = self.equity(price)
        return (self.position_units * price / eq) if eq else 0.0

    def rebalance_to(self, target_fraction: float, price: float, when: str) -> dict | None:
        """Move toward `target_fraction` of current equity at `price`.

        Returns a trade record dict if a trade happened, else None.
        """
        target_fraction = max(-1.0, min(1.0, target_fraction))
        eq = self.equity(price)
        if eq <= 0:
            return None
        desired_units = target_fraction * eq / price
        delta_units = desired_units - self.position_units
        trade_notional = abs(delta_units) * price

        if trade_notional < _MIN_TRADE_FRAC * eq:
            return None  # too small to bother

        cost = trade_notional * self.cost_rate
        self.cash -= delta_units * price   # buy (delta>0) spends cash; sell adds
        self.cash -= cost
        self.position_units = desired_units

        rec = {
            "time": when,
            "side": "buy" if delta_units > 0 else "sell",
            "price": price,
            "delta_units": delta_units,
            "target_fraction": target_fraction,
            "notional": trade_notional,
            "cost": cost,
            "cash_after": self.cash,
            "position_units_after": self.position_units,
            "equity_after": self.equity(price),
        }
        self.trades.append(rec)
        return rec

    def snapshot(self, when: str, price: float) -> None:
        self.equity_curve.append({
            "time": when,
            "price": price,
            "position_units": self.position_units,
            "position_fraction": self.position_fraction(price),
            "equity": self.equity(price),
        })

    # --- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "cost_rate": self.cost_rate,
            "cash": self.cash,
            "position_units": self.position_units,
            "initial_equity": self.initial_equity,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperPortfolio":
        return cls(
            cost_rate=d["cost_rate"],
            cash=d["cash"],
            position_units=d["position_units"],
            initial_equity=d.get("initial_equity", d["cash"]),
            trades=d.get("trades", []),
            equity_curve=d.get("equity_curve", []),
        )
