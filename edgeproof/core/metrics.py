"""Performance metrics, with overfitting-aware Sharpe ratios.

The headline guard here is the **Deflated Sharpe Ratio** (Bailey & Lopez de
Prado, 2014). A high backtest Sharpe is meaningless if you tried 200 variants;
DSR asks: "given that I ran N trials, what's the probability this Sharpe is
truly > 0 rather than the luckiest of N noisy draws?"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

_EULER_GAMMA = 0.5772156649015329

# bars per year, for annualising Sharpe / returns
_BARS_PER_YEAR = {
    "1m": 525_600, "3m": 175_200, "5m": 105_120, "15m": 35_040, "30m": 17_520,
    "1h": 8_760, "2h": 4_380, "4h": 2_190, "6h": 1_460, "8h": 1_095,
    "12h": 730, "1d": 365, "3d": 121.7, "1w": 52,
}


def annualization_factor(interval: str) -> float:
    if interval not in _BARS_PER_YEAR:
        raise ValueError(f"no annualization factor for interval {interval!r}")
    return float(_BARS_PER_YEAR[interval])


def _per_bar_sharpe(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(returns.mean() / sd)


def probabilistic_sharpe_ratio(
    returns: pd.Series, sr_benchmark_per_bar: float = 0.0
) -> float:
    """P(true per-bar Sharpe > benchmark), adjusting for skew & fat tails.

    Returns a probability in [0, 1]. Uses the non-annualised (per-bar) Sharpe.
    """
    r = pd.Series(returns).dropna().to_numpy()
    T = len(r)
    if T < 3:
        return float("nan")
    sr = _per_bar_sharpe(r)
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # non-excess
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2))
    z = (sr - sr_benchmark_per_bar) * math.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe_per_bar(n_trials: int, sr_trials_std: float) -> float:
    """Expected maximum per-bar Sharpe across N independent random trials.

    This is the SR* you must beat to claim a real edge after searching N
    variants. Grows with both the number of trials and how dispersed their
    Sharpes are.
    """
    if n_trials < 2 or sr_trials_std <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sr_trials_std * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, sr_trials_std: float
) -> float:
    """Probability the strategy's true Sharpe beats the best-of-N-luck threshold.

    Needs the number of variants tried (`n_trials`) and the dispersion of their
    per-bar Sharpes (`sr_trials_std`). With a single trial this collapses to the
    PSR-vs-0; in that case prefer `probabilistic_sharpe_ratio`.
    """
    sr_star = expected_max_sharpe_per_bar(n_trials, sr_trials_std)
    return probabilistic_sharpe_ratio(returns, sr_benchmark_per_bar=sr_star)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


@dataclass
class Report:
    bars: int
    interval: str
    total_return: float
    cagr: float
    ann_return: float
    ann_volatility: float
    sharpe: float            # annualised, net of costs
    max_drawdown: float
    win_rate: float
    n_trades: int
    total_cost_drag: float   # cumulative fraction of equity lost to costs
    psr_vs_zero: float       # P(true Sharpe > 0)
    deflated_sharpe: float | None  # P(true Sharpe > best-of-N luck), if n_trials>1
    benchmark_total_return: float
    benchmark_sharpe: float

    def as_dict(self) -> dict:
        return asdict(self)


def performance_report(
    result,
    *,
    n_trials: int = 1,
    sr_trials_std: float = 0.0,
) -> Report:
    net = result.net_returns.dropna()
    af = annualization_factor(result.interval)
    sr_bar = _per_bar_sharpe(net.to_numpy())
    ann_sharpe = sr_bar * math.sqrt(af)

    equity = result.equity
    total_ret = float(equity.iloc[-1] - 1.0)
    n_bars = len(net)
    years = n_bars / af if af else float("nan")
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    # win rate over bars with an active position
    active = net[result.position.reindex(net.index).abs() > 1e-9]
    win_rate = float((active > 0).mean()) if len(active) else float("nan")

    bench_net = (result.benchmark_equity.pct_change().fillna(0.0)).to_numpy()
    bench_sharpe = _per_bar_sharpe(bench_net) * math.sqrt(af)

    dsr = None
    if n_trials > 1 and sr_trials_std > 0:
        dsr = deflated_sharpe_ratio(net, n_trials, sr_trials_std)

    return Report(
        bars=n_bars,
        interval=result.interval,
        total_return=total_ret,
        cagr=cagr,
        ann_return=float(net.mean() * af),
        ann_volatility=float(net.std(ddof=1) * math.sqrt(af)),
        sharpe=ann_sharpe,
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate,
        n_trades=result.n_trades,
        total_cost_drag=float(result.costs.sum()),
        psr_vs_zero=probabilistic_sharpe_ratio(net, 0.0),
        deflated_sharpe=dsr,
        benchmark_total_return=float(result.benchmark_equity.iloc[-1] - 1.0),
        benchmark_sharpe=bench_sharpe,
    )
