import numpy as np
import pandas as pd

from edgeproof.core import walk_forward, CostModel, performance_report
from edgeproof.core.walkforward import _make_splits
from edgeproof.strategies import MACrossStrategy


def _synthetic(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"close": close, "quote_volume": rng.uniform(1e6, 5e6, n)}, index=idx)


def test_splits_are_contiguous_and_non_overlapping():
    splits = _make_splits(3000, train_bars=1500, test_bars=400, anchored=False)
    assert len(splits) >= 3
    # test windows tile forward without gaps or overlaps
    for (a, b) in zip(splits, splits[1:]):
        assert a[3] == b[2], "test windows must be contiguous"
    # every test window sits strictly after its own train window
    for (tr_s, tr_e, te_s, te_e) in splits:
        assert tr_e == te_s
        assert te_s > tr_s


def test_walk_forward_runs_and_scores_only_out_of_sample():
    data = _synthetic()
    grid = [{"fast": f, "slow": s} for f in (10, 20, 30) for s in (50, 100) if f < s]
    wf = walk_forward(data, MACrossStrategy, grid, interval="1h",
                      cost_model=CostModel(), train_bars=1500, test_bars=400)

    # OOS span starts only after the first training window
    assert wf.oos_result.net_returns.index[0] >= data.index[1500]
    assert wf.n_trials == len(grid)
    assert len(wf.splits) >= 3

    rep = performance_report(wf.oos_result, n_trials=wf.n_trials, sr_trials_std=wf.sr_trials_std)
    assert 0.0 <= rep.psr_vs_zero <= 1.0
    # on pure random data there should be no real out-of-sample edge
    assert rep.deflated_sharpe is None or rep.deflated_sharpe < 0.95


def test_anchored_grows_train_window():
    splits = _make_splits(3000, train_bars=1000, test_bars=400, anchored=True)
    # anchored train always starts at 0
    assert all(s[0] == 0 for s in splits)
