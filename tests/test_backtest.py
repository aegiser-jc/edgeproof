"""Correctness tests — the most important is that the engine cannot look ahead."""
import numpy as np
import pandas as pd
import pytest

from edgeproof.core import run_backtest, CostModel, performance_report
from edgeproof.core.metrics import (
    probabilistic_sharpe_ratio,
    expected_max_sharpe_per_bar,
)


def _synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"close": close, "quote_volume": rng.uniform(1e6, 5e6, n)}, index=idx
    )


def test_no_lookahead_same_bar_cheat_is_neutralised():
    """A signal that 'knows' the current bar's move must NOT earn that bar."""
    data = _synthetic()
    bar_ret = data["close"].pct_change().fillna(0.0)
    cheat = np.sign(bar_ret)  # cheating: uses bar t's own return at bar t

    res = run_backtest(data, cheat, cost_model=CostModel(0, 0, 0))

    # Under look-ahead, gross per-bar return would be |bar_ret| (always positive).
    perfect = bar_ret.abs().sum()
    assert res.gross_returns.sum() < perfect * 0.6, "engine is leaking future data!"

    # Held position must be exactly the target shifted by one bar.
    expected_held = cheat.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(
        res.position, expected_held, check_names=False
    )


def test_costs_reduce_returns_and_scale_with_turnover():
    data = _synthetic()
    flip = pd.Series(np.where(np.arange(len(data)) % 2 == 0, 1.0, -1.0), index=data.index)

    free = run_backtest(data, flip, cost_model=CostModel(0, 0, 0))
    pricey = run_backtest(data, flip, cost_model=CostModel(fee_bps=50, slippage_bps=0, half_spread_bps=0))

    assert pricey.equity.iloc[-1] < free.equity.iloc[-1]
    assert pricey.costs.sum() > 0
    # flipping every bar => high trade count
    assert pricey.n_trades > len(data) * 0.4


def test_buy_hold_matches_price_path_minus_one_cost():
    data = _synthetic()
    hold = pd.Series(1.0, index=data.index)
    res = run_backtest(data, hold, cost_model=CostModel(0, 0, 0))
    # strategy fully long with no costs should track price return closely
    price_total = data["close"].iloc[-1] / data["close"].iloc[0] - 1.0
    assert abs(res.equity.iloc[-1] - 1.0 - price_total) < 1e-6


def test_capacity_warning_triggers_when_oversized():
    data = _synthetic()
    data["quote_volume"] = 1_000.0  # tiny market
    flip = pd.Series(np.where(np.arange(len(data)) % 2 == 0, 1.0, 0.0), index=data.index)
    res = run_backtest(data, flip, capacity_usd=1_000_000, capacity_volume_frac=0.01)
    assert res.capacity_warnings, "expected a capacity warning on a tiny market"


def test_psr_and_expected_max_sharpe_are_sane():
    rng = np.random.default_rng(1)
    good = pd.Series(rng.normal(0.005, 0.01, 1000))  # clearly positive Sharpe
    assert probabilistic_sharpe_ratio(good, 0.0) > 0.99

    noise = pd.Series(rng.normal(0.0, 0.01, 1000))
    assert probabilistic_sharpe_ratio(noise, 0.0) < 0.95

    # more trials => higher bar to clear
    sr_few = expected_max_sharpe_per_bar(5, 0.05)
    sr_many = expected_max_sharpe_per_bar(200, 0.05)
    assert sr_many > sr_few > 0


def test_performance_report_runs():
    data = _synthetic()
    sig = (data["close"].rolling(10).mean() > data["close"].rolling(30).mean()).astype(float)
    res = run_backtest(data, sig, interval="1h")
    rep = performance_report(res, n_trials=12, sr_trials_std=0.03)
    assert rep.deflated_sharpe is not None
    assert 0.0 <= rep.psr_vs_zero <= 1.0
    assert rep.n_trades > 0
