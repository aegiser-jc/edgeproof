# EdgeProof

![Python](https://img.shields.io/badge/python-3.11-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Prove a trading strategy's edge is real — *net of costs* — instead of fooling yourself with a pretty backtest.**

Most retail systematic strategies lose, and the usual reasons are self-inflicted:
backtests that peek at the future, ignore trading costs, or are simply the
luckiest of many variants tried. EdgeProof is a small, honest research harness
that makes each of those failure modes hard to commit by accident.

| Failure mode | What EdgeProof does about it |
|---|---|
| Look-ahead bias | Signals computed on bar *t* can only execute on *t+1* — enforced by construction. A unit test proves a "cheating" same-bar signal earns nothing. |
| Gross numbers hide reality | Every backtest charges fee + half-spread + slippage; only **net** results are reported. |
| Overfitting / best-of-N luck | **Deflated Sharpe Ratio** penalises how many variants you tried; **walk-forward** scores parameters out-of-sample. |
| "Edge" only at untradeable size | Position size is checked against per-bar traded volume. |

EdgeProof is the *judge*, not the trader. Bring a strategy idea; it tells you,
honestly, whether the edge survives reality.

## Install & run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# single backtest (free Binance data, no API key needed)
python -m edgeproof.cli backtest --symbol BTCUSDT --interval 1h --strategy ma_cross --fast 20 --slow 50

# grid-search and DEFLATE the best Sharpe for the variants tried
python -m edgeproof.cli scan --symbol BTCUSDT --interval 1h --strategy vol_shock_drift

# walk-forward: pick params in-sample, score out-of-sample (the strongest guard)
python -m edgeproof.cli walkforward --symbol BTCUSDT --interval 1h --bars 8000 --train-bars 3000 --test-bars 800

# live paper trading on fake money against real prices (idempotent, state-persisted)
python -m edgeproof.cli paper-run    --symbol BTCUSDT --interval 1h --strategy ma_cross --poll 60
python -m edgeproof.cli paper-status --symbol BTCUSDT --interval 1h --strategy ma_cross
```

A Deflated Sharpe near 95%+ that also beats buy-and-hold net of costs is the only
thing worth paper-trading. On a textbook moving-average cross it is correctly
near zero — which is the point.

## Layout

```
edgeproof/
  data/binance.py     free OHLCV via Binance public REST, parquet-cached
  core/strategy.py    Strategy interface (target position in [-1, 1])
  core/backtest.py    causal, cost-charging engine + capacity check
  core/metrics.py     Sharpe, drawdown, PSR, Deflated Sharpe
  core/walkforward.py rolling / anchored out-of-sample validation
  strategies/         ma_cross, buy_hold, vol_shock_drift
  paper/              live fake-money portfolio + engine
  cli.py              backtest / scan / walkforward / paper-* commands
tests/                no-look-ahead, cost, metric & paper correctness tests
```

## Tests

```bash
pip install pytest && python -m pytest -q
```

## Disclaimer

For research and education. Not financial advice. Markets are risky; a passing
backtest is necessary but never sufficient.

## License

MIT — see [LICENSE](LICENSE).
