# EdgeProof

**Prove a trading edge is real *net of costs* — not a backtest illusion.**

EdgeProof is a research/backtesting harness for systematic crypto strategies.
It is built around an uncomfortable, well-documented fact: retail systematic
trading almost always loses, and the #1 reason is **self-deception** — backtests that look great because they
peek at the future, ignore trading costs, or are the luckiest of many variants
tried.

EdgeProof makes each of those failure modes structurally hard:

| Failure mode (from the literature) | What EdgeProof does |
|---|---|
| Look-ahead bias inflates backtests | Signals at bar *t* are executed at *t+1* by construction (positions shifted one bar). A test proves a same-bar "cheating" signal earns nothing. |
| Gross returns hide real-world frictions | Every backtest charges fee + half-spread + slippage. Only **net** numbers are reported. |
| Overfitting / best-of-N luck | **Deflated Sharpe Ratio** penalises how many variants you tried. |
| "Edge" only exists at untradeable size | Capacity check vs. per-bar traded volume. |

> EdgeProof is the *judge*, not the trader. Use Claude Code to generate and
> iterate strategy ideas; let EdgeProof tell you, honestly, whether any of them
> survive reality.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r ../requirements.txt

# single backtest (free Binance data, no API key)
python -m edgeproof.cli backtest --symbol BTCUSDT --interval 1h --bars 4000 \
    --strategy ma_cross --fast 20 --slow 50 --capital 5000

# grid-search and DEFLATE the best Sharpe for the variants tried
python -m edgeproof.cli scan --symbol BTCUSDT --interval 1h --bars 4000

# walk-forward: choose params in-sample, score out-of-sample, rolling (strongest guard)
python -m edgeproof.cli walkforward --symbol BTCUSDT --interval 1h --bars 8000 \
    --train-bars 3000 --test-bars 800

# paper trading: run live on fake money against real prices (24/7 box)
python -m edgeproof.cli paper-run    --symbol BTCUSDT --interval 1h --strategy ma_cross --poll 60
python -m edgeproof.cli paper-status --symbol BTCUSDT --interval 1h --strategy ma_cross
python -m edgeproof.cli paper-reset  --symbol BTCUSDT --interval 1h --strategy ma_cross
```

**Running paper trading 24/7.** `paper-run` ticks are idempotent (they act only
when a new bar closes) and persist state every tick, so any of these works:
a background process (`nohup ... &` / tmux / systemd), or a cron job calling
`paper-run --once` on each bar boundary. Check in any time with `paper-status`.

The walk-forward per-fold table shows in-sample vs out-of-sample Sharpe side by
side. A strategy whose IS Sharpe looks great but whose OOS Sharpe collapses (or
flips sign) is overfit — the textbook MA-cross does exactly this, which is the
correct, honest result.

Reading the verdict: a Deflated Sharpe near **95%+** while also beating buy&hold
net of costs is the only thing worth paper-trading. Anything less is most likely
noise. (On a textbook MA-cross it is correctly ~16% — i.e. nothing.)

## Layout

```
edgeproof/
  data/binance.py     free OHLCV via Binance public REST, parquet-cached
  core/strategy.py    Strategy interface (target position in [-1,1])
  core/backtest.py    causal, cost-charging engine + capacity check
  core/metrics.py     Sharpe, drawdown, PSR, Deflated Sharpe
  strategies/         ma_cross, buy_hold, vol_shock_drift (post-news-drift proxy)
  paper/              live fake-money portfolio + engine (idempotent, persisted)
  cli.py              backtest / scan / walkforward / paper-* commands (rich output)
tests/                no-look-ahead + cost + metric + paper correctness tests (19)
```

## Roadmap

- [x] Causal backtest engine, realistic costs, Deflated Sharpe, capacity check
- [x] Free Binance data + caching, CLI, test suite
- [x] **Walk-forward / out-of-sample split** (train params on A, score on B; rolling or anchored)
- [ ] **Paper-trading loop** — run live on the 24/7 box against real-time prices, fake money, full trade log
- [ ] More data sources (ccxt → multi-exchange) and asset universes
- [x] First theory-backed strategy: `vol_shock_drift` (post-news-drift proxy) — tested, did NOT survive walk-forward (spectacular in-sample Sharpe, negative out-of-sample); the harness correctly rejected a scan-level false positive
- [x] **Paper-trading loop** — live on the 24/7 box against real prices, fake money, full trade log; idempotent ticks + persisted state (`paper-run` / `paper-status` / `paper-reset`)
- [ ] Strategy plugins + a small library of non-trivial ideas to test
- [ ] Web dashboard / shareable reports → the productization path (sell to systematic retail traders; technical buyers, low marketing burden)

## Honest stance

The research is clear that net-positive retail capture is rarely demonstrated.
The realistic win here is **two-in-one**: if no strategy survives EdgeProof
(the likely outcome), you have lost only time and gained a rigorous, reusable
harness that itself is a sellable tool for technical traders. If something does
survive walk-forward + paper trading, that is the bonus — pursued with small,
bounded risk.
