"""Walk-forward analysis — the strongest cheap guard against overfitting.

The lie a plain backtest tells: "I picked the best parameters, look how well
they did." But you picked them *with knowledge of the whole period*. Walk-forward
removes that cheat:

    for each fold:
        choose the best parameters using ONLY the in-sample (train) window
        score those parameters on the next, unseen out-of-sample (test) window
        roll forward

The concatenated out-of-sample returns are an honest estimate of what you would
have earned choosing parameters the way you actually would in real time — with
no knowledge of the future. A strategy that shines in-sample and dies
out-of-sample is overfit, and this is where you catch it.

Signal generation is causal (signal[t] uses only data <= t), so we generate each
variant's signals over the full series once and then *select* on the train slice
and *score* on the test slice. The only thing forbidden — selecting parameters
using test-window performance — is exactly what this code refuses to do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import CostModel, BacktestResult, run_backtest


def _per_bar_sharpe(returns: pd.Series) -> float:
    a = np.asarray(returns.dropna())
    if len(a) < 2:
        return 0.0
    sd = a.std(ddof=1)
    return 0.0 if sd == 0 or not np.isfinite(sd) else float(a.mean() / sd)


@dataclass
class WalkForwardResult:
    oos_result: BacktestResult        # synthetic, concatenated out-of-sample
    splits: list = field(default_factory=list)  # per-fold diagnostics
    n_trials: int = 1                 # variants searched at each selection
    sr_trials_std: float = 0.0        # dispersion of variant Sharpes (for DSR)
    mean_is_sharpe: float = float("nan")   # avg in-sample Sharpe of chosen params
    oos_sharpe: float = float("nan")       # realized out-of-sample (annualized handled by report)


def _make_splits(n: int, train_bars: int, test_bars: int, anchored: bool):
    """Yield (train_start, train_end, test_start, test_end) index ranges."""
    splits = []
    test_start = train_bars
    while test_start + test_bars <= n:
        test_end = test_start + test_bars
        train_start = 0 if anchored else test_start - train_bars
        splits.append((train_start, test_start, test_start, test_end))
        test_start += test_bars
    # absorb a trailing remainder into a final fold if it's at least half a window
    if test_start < n and (n - test_start) >= test_bars // 2:
        train_start = 0 if anchored else test_start - train_bars
        train_start = max(0, train_start)
        splits.append((train_start, test_start, test_start, n))
    return splits


def walk_forward(
    data: pd.DataFrame,
    strategy_cls,
    param_grid: list[dict],
    *,
    interval: str = "1h",
    cost_model: CostModel | None = None,
    train_bars: int = 1500,
    test_bars: int = 400,
    anchored: bool = False,
) -> WalkForwardResult:
    cost_model = cost_model or CostModel()
    n = len(data)
    splits = _make_splits(n, train_bars, test_bars, anchored)
    if not splits:
        raise ValueError(
            f"not enough data ({n} bars) for train={train_bars}+test={test_bars}"
        )

    # Run each variant once over the full series (signals are causal).
    variants = []  # (params, BacktestResult, full_sample_per_bar_sharpe)
    for params in param_grid:
        strat = strategy_cls(**params)
        res = run_backtest(data, strat.generate_signals(data),
                           interval=interval, cost_model=cost_model)
        variants.append((params, res, _per_bar_sharpe(res.net_returns)))

    sr_trials_std = float(np.std([v[2] for v in variants], ddof=1)) if len(variants) > 1 else 0.0

    # Per fold: pick best on train, harvest on test.
    oos_net, oos_gross, oos_pos, oos_turn, oos_cost = [], [], [], [], []
    fold_diag = []
    is_sharpes = []
    prev_last_pos = 0.0
    rate = cost_model.rate

    for (tr_s, tr_e, te_s, te_e) in splits:
        # selection: best in-sample per-bar Sharpe
        best = None
        for params, res, _ in variants:
            train_ret = res.net_returns.iloc[tr_s:tr_e]
            sr = _per_bar_sharpe(train_ret)
            if best is None or sr > best[0]:
                best = (sr, params, res)
        is_sr, best_params, best_res = best
        is_sharpes.append(is_sr)

        # score: that variant's behaviour over the unseen test window
        net = best_res.net_returns.iloc[te_s:te_e].copy()
        pos = best_res.position.iloc[te_s:te_e].copy()
        turn = best_res.turnover.iloc[te_s:te_e].copy()
        gross = best_res.gross_returns.iloc[te_s:te_e].copy()
        cost = best_res.costs.iloc[te_s:te_e].copy()

        # honesty: charge the switch from the previous fold's final position
        if len(pos):
            boundary_turn = abs(float(pos.iloc[0]) - prev_last_pos)
            extra = boundary_turn * rate
            turn.iloc[0] = turn.iloc[0] + boundary_turn
            cost.iloc[0] = cost.iloc[0] + extra
            net.iloc[0] = net.iloc[0] - extra
            prev_last_pos = float(pos.iloc[-1])

        oos_net.append(net); oos_gross.append(gross)
        oos_pos.append(pos); oos_turn.append(turn); oos_cost.append(cost)
        fold_diag.append({
            "train": (tr_s, tr_e), "test": (te_s, te_e),
            "params": best_params, "is_sharpe": is_sr,
            "oos_sharpe": _per_bar_sharpe(net),
        })

    net = pd.concat(oos_net); gross = pd.concat(oos_gross)
    pos = pd.concat(oos_pos); turn = pd.concat(oos_turn); cost = pd.concat(oos_cost)
    equity = (1.0 + net).cumprod()

    # buy & hold over the same out-of-sample span
    oos_index = net.index
    close = data["close"].reindex(oos_index)
    bar_ret = close.pct_change().fillna(0.0)
    bh_cost = pd.Series(0.0, index=oos_index)
    if len(bh_cost) > 1:
        bh_cost.iloc[1] = rate
    bench = (1.0 + bar_ret - bh_cost).cumprod()

    synthetic = BacktestResult(
        equity=equity, net_returns=net, gross_returns=gross, position=pos,
        turnover=turn, costs=cost, benchmark_equity=bench, interval=interval,
        cost_model=cost_model, capacity_warnings=[],
    )

    return WalkForwardResult(
        oos_result=synthetic,
        splits=fold_diag,
        n_trials=len(variants),
        sr_trials_std=sr_trials_std,
        mean_is_sharpe=float(np.mean(is_sharpes)) if is_sharpes else float("nan"),
        oos_sharpe=_per_bar_sharpe(net),
    )
