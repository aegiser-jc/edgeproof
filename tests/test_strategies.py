import numpy as np
import pandas as pd

from edgeproof.strategies import VolumeShockDriftStrategy


def _data_with_event(n=300, event_at=150, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    volume = rng.uniform(900, 1100, n)
    # inject a clear up-shock: big return + huge volume at `event_at`
    close[event_at:] *= 1.05            # +5% jump that persists
    volume[event_at] = 50_000           # volume spike
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"close": close, "volume": volume,
                         "quote_volume": volume * close}, index=idx), event_at


def test_drift_fires_on_shock_and_holds_for_hold_bars():
    data, ev = _data_with_event()
    strat = VolumeShockDriftStrategy(vol_window=96, vol_z=3.0, ret_thresh=0.01,
                                     hold_bars=6, mode="momentum", event_side="both")
    sig = strat.generate_signals(data)
    # the up-shock should produce a long signal at the event bar...
    assert sig.iloc[ev] == 1.0
    # ...held for hold_bars bars total, then flat
    assert (sig.iloc[ev:ev + 6] == 1.0).all()
    assert sig.iloc[ev + 6] == 0.0


def test_drift_no_signal_during_warmup():
    data, _ = _data_with_event()
    strat = VolumeShockDriftStrategy(vol_window=96)
    sig = strat.generate_signals(data)
    assert (sig.iloc[:96] == 0.0).all()


def test_drift_event_side_down_ignores_up_shocks():
    data, ev = _data_with_event()  # this is an UP shock
    strat = VolumeShockDriftStrategy(vol_z=3.0, ret_thresh=0.01, event_side="down")
    sig = strat.generate_signals(data)
    assert sig.iloc[ev] == 0.0  # up shock ignored when only trading down events


def test_drift_reversal_flips_direction():
    data, ev = _data_with_event()
    strat = VolumeShockDriftStrategy(vol_z=3.0, ret_thresh=0.01, mode="reversal")
    sig = strat.generate_signals(data)
    assert sig.iloc[ev] == -1.0  # bet against an up shock


def test_drift_long_only_clamps_shorts():
    data, ev = _data_with_event()
    strat = VolumeShockDriftStrategy(vol_z=3.0, ret_thresh=0.01, mode="reversal",
                                     long_only=True)
    sig = strat.generate_signals(data)
    assert (sig >= 0.0).all()  # no shorts when long_only
