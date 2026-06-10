"""EdgeProof command line.

    python -m edgeproof.cli backtest --symbol BTCUSDT --interval 1h --bars 4000 \
        --strategy ma_cross --fast 20 --slow 50 --capital 5000

    python -m edgeproof.cli scan --symbol BTCUSDT --interval 1h \
        --strategy vol_shock_drift --event-side down

    python -m edgeproof.cli walkforward --symbol BTCUSDT --interval 1h --bars 8000 \
        --strategy vol_shock_drift --train-bars 3000 --test-bars 800
"""
from __future__ import annotations

import argparse
import math

import numpy as np
from rich.console import Console
from rich.table import Table

from .data import fetch_klines
from .core import run_backtest, CostModel, performance_report, walk_forward
from .core.metrics import annualization_factor
from .strategies import REGISTRY
from .paper import PaperTrader, PaperPortfolio

console = Console()


# ---- strategy parameter plumbing -------------------------------------------

def _strategy_kwargs(args) -> dict:
    """Params for a single backtest run."""
    if args.strategy == "ma_cross":
        return dict(fast=args.fast, slow=args.slow, allow_short=args.allow_short)
    if args.strategy == "vol_shock_drift":
        return dict(vol_window=args.vol_window, vol_z=args.vol_z,
                    ret_thresh=args.ret_thresh, hold_bars=args.hold,
                    mode=args.mode, event_side=args.event_side, long_only=args.long_only)
    return {}


def _grid(args):
    """Variant grid for scan / walk-forward. Fixed (non-swept) params come from args."""
    if args.strategy == "ma_cross":
        return [
            {"fast": f, "slow": s, "allow_short": args.allow_short}
            for f in (5, 10, 20, 30, 50) for s in (50, 100, 150, 200) if f < s
        ]
    if args.strategy == "vol_shock_drift":
        return [
            {"vol_window": args.vol_window, "vol_z": z, "ret_thresh": rt,
             "hold_bars": h, "mode": args.mode, "event_side": args.event_side,
             "long_only": args.long_only}
            for z in (2.0, 3.0, 4.0)
            for rt in (0.005, 0.01, 0.02)
            for h in (3, 6, 12)
        ]
    return None


def _params_str(strategy: str, p: dict) -> str:
    if strategy == "ma_cross":
        return f"fast={p['fast']}, slow={p['slow']}"
    if strategy == "vol_shock_drift":
        return f"z={p['vol_z']}, rt={p['ret_thresh']}, h={p['hold_bars']}"
    return ", ".join(f"{k}={v}" for k, v in p.items())


# ---- formatting helpers -----------------------------------------------------

def _per_bar_sharpe(net) -> float:
    a = np.asarray(net.dropna())
    sd = a.std(ddof=1)
    return 0.0 if sd == 0 or not np.isfinite(sd) else float(a.mean() / sd)


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x*100:,.2f}%"


def _num(x, nd: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{nd}f}"


def _cost_model(args) -> CostModel:
    return CostModel(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                     half_spread_bps=args.spread_bps)


def _render_report(rep, *, symbol, strategy_repr):
    t = Table(title=f"EdgeProof — {symbol} — {strategy_repr}", title_style="bold cyan")
    t.add_column("metric", style="bold")
    t.add_column("strategy", justify="right")
    t.add_column("buy & hold", justify="right", style="dim")

    t.add_row("bars / interval", f"{rep.bars} / {rep.interval}", "")
    t.add_row("total return (net)", _pct(rep.total_return), _pct(rep.benchmark_total_return))
    t.add_row("CAGR", _pct(rep.cagr), "")
    t.add_row("ann. volatility", _pct(rep.ann_volatility), "")
    t.add_row("Sharpe (net, ann.)", _num(rep.sharpe), _num(rep.benchmark_sharpe))
    t.add_row("max drawdown", _pct(rep.max_drawdown), "")
    t.add_row("win rate (active bars)", _pct(rep.win_rate), "")
    t.add_row("# trades", str(rep.n_trades), "")
    t.add_row("cost drag (total)", _pct(rep.total_cost_drag), "")
    t.add_row("PSR  P(Sharpe>0)", _pct(rep.psr_vs_zero), "")
    t.add_row("Deflated Sharpe", _pct(rep.deflated_sharpe), "")
    console.print(t)


def _verdict(rep) -> None:
    beats_bh = rep.sharpe > rep.benchmark_sharpe
    dsr = rep.deflated_sharpe
    if dsr is not None:
        if dsr >= 0.95 and beats_bh:
            msg = "[green]Survives overfitting deflation AND beats buy&hold net of costs. Worth paper-trading.[/]"
        elif dsr < 0.90:
            msg = "[red]Deflated Sharpe is weak — most likely the best-of-luck from the search, not a real edge.[/]"
        else:
            msg = "[yellow]Borderline. Re-test out-of-sample before believing it.[/]"
    else:
        if rep.psr_vs_zero >= 0.95 and beats_bh:
            msg = "[yellow]Single run looks positive, but run `scan`/`wf` to deflate for the variants you'll try.[/]"
        else:
            msg = "[red]No convincing edge net of costs.[/]"
    console.print(f"\n[bold]Verdict:[/] {msg}")


# ---- commands ---------------------------------------------------------------

def cmd_backtest(args):
    data = fetch_klines(args.symbol, args.interval, args.bars, force_refresh=args.refresh)
    strat = REGISTRY[args.strategy](**_strategy_kwargs(args))
    result = run_backtest(
        data, strat.generate_signals(data), interval=args.interval,
        cost_model=_cost_model(args), capacity_usd=args.capital,
    )
    rep = performance_report(result)
    _render_report(rep, symbol=args.symbol, strategy_repr=repr(strat))
    for w in result.capacity_warnings:
        console.print(f"[yellow]⚠ capacity:[/] {w}")
    _verdict(rep)


def cmd_scan(args):
    grid = _grid(args)
    if grid is None:
        console.print(f"[red]scan not supported for --strategy {args.strategy}[/]")
        return
    data = fetch_klines(args.symbol, args.interval, args.bars, force_refresh=args.refresh)
    cost = _cost_model(args)
    af = annualization_factor(args.interval)

    rows = []
    for params in grid:
        strat = REGISTRY[args.strategy](**params)
        result = run_backtest(data, strat.generate_signals(data),
                              interval=args.interval, cost_model=cost)
        rows.append((params, _per_bar_sharpe(result.net_returns), result))

    sr_bars = np.array([r[1] for r in rows])
    sr_trials_std = float(sr_bars.std(ddof=1)) if len(rows) > 1 else 0.0
    n_trials = len(rows)
    rows.sort(key=lambda r: r[1], reverse=True)

    t = Table(title=f"EdgeProof scan — {args.symbol} {args.interval} — "
                    f"{args.strategy} — {n_trials} variants tried", title_style="bold cyan")
    t.add_column("params"); t.add_column("Sharpe(net,ann)", justify="right")
    t.add_column("total ret", justify="right"); t.add_column("# trades", justify="right")
    for params, sr_bar, result in rows[:10]:
        rep = performance_report(result)
        t.add_row(_params_str(args.strategy, params), _num(sr_bar * math.sqrt(af)),
                  _pct(rep.total_return), str(rep.n_trades))
    console.print(t)

    best_params, _, best_res = rows[0]
    best_rep = performance_report(best_res, n_trials=n_trials, sr_trials_std=sr_trials_std)
    console.print(f"\n[bold]Best variant:[/] {_params_str(args.strategy, best_params)} — "
                  f"but it is the best of [bold]{n_trials}[/] tries.")
    _render_report(best_rep, symbol=args.symbol,
                   strategy_repr=f"{args.strategy}({_params_str(args.strategy, best_params)}) "
                                 f"[best of {n_trials}]")
    _verdict(best_rep)


def cmd_walkforward(args):
    grid = _grid(args)
    if grid is None:
        console.print(f"[red]walkforward not supported for --strategy {args.strategy}[/]")
        return
    data = fetch_klines(args.symbol, args.interval, args.bars, force_refresh=args.refresh)

    wf = walk_forward(
        data, REGISTRY[args.strategy], grid,
        interval=args.interval, cost_model=_cost_model(args),
        train_bars=args.train_bars, test_bars=args.test_bars, anchored=args.anchored,
    )
    af = annualization_factor(args.interval)

    t = Table(title=f"EdgeProof walk-forward — {args.symbol} {args.interval} — {args.strategy} — "
                    f"{len(wf.splits)} folds, {wf.n_trials} variants/selection "
                    f"({'anchored' if args.anchored else 'rolling'})", title_style="bold cyan")
    t.add_column("fold"); t.add_column("chosen params")
    t.add_column("IS Sharpe", justify="right"); t.add_column("OOS Sharpe", justify="right")
    for i, fold in enumerate(wf.splits, 1):
        oos = fold["oos_sharpe"] * math.sqrt(af)
        style = "green" if oos > 0 else "red"
        t.add_row(str(i), _params_str(args.strategy, fold["params"]),
                  _num(fold["is_sharpe"] * math.sqrt(af)), f"[{style}]{_num(oos)}[/]")
    console.print(t)

    is_ann = wf.mean_is_sharpe * math.sqrt(af)
    oos_ann = wf.oos_sharpe * math.sqrt(af)
    console.print(
        f"\n[bold]In-sample avg Sharpe:[/] {_num(is_ann)}   "
        f"[bold]→ Out-of-sample Sharpe:[/] {_num(oos_ann)}   "
        f"([{'red' if oos_ann < is_ann * 0.5 else 'yellow'}]"
        f"{_pct((oos_ann - is_ann) / is_ann) if is_ann else '—'} change[/])"
    )

    rep = performance_report(wf.oos_result, n_trials=wf.n_trials, sr_trials_std=wf.sr_trials_std)
    _render_report(rep, symbol=args.symbol,
                   strategy_repr=f"{args.strategy} [walk-forward OOS, {len(wf.splits)} folds]")
    if oos_ann < is_ann * 0.5:
        console.print("[red]\nLarge in-sample → out-of-sample decay = classic overfitting. "
                      "The 'edge' was curve-fit to the past.[/]")
    _verdict(rep)


def _get_trader(args) -> PaperTrader:
    """Load an existing paper run, or create one from CLI args."""
    name = args.name or f"{args.symbol}_{args.interval}_{args.strategy}"
    if PaperTrader.exists(name):
        trader = PaperTrader.load(name)
        if any([args.symbol != trader.symbol, args.interval != trader.interval,
                args.strategy != trader.strategy]) and not getattr(args, "_status", False):
            console.print(f"[yellow]Note:[/] reusing existing run '{name}' "
                          f"({trader.symbol} {trader.interval} {trader.strategy}); "
                          f"`paper-reset` to start fresh.")
        return trader
    portfolio = PaperPortfolio(cost_rate=_cost_model(args).rate,
                               cash=args.equity, initial_equity=args.equity)
    return PaperTrader(args.symbol, args.interval, args.strategy,
                       _strategy_kwargs(args), portfolio, name=name)


def _print_tick(status):
    if not status.get("acted"):
        console.print(f"[dim]· no action ({status.get('reason')})[/]")
        return
    eq = status["equity"]
    line = (f"[cyan]{status['bar']}[/]  price={status['price']:,.4f}  "
            f"target={status['target_fraction']:+.2f}  equity=${eq:,.2f}")
    if status["traded"]:
        tr = status["trade"]
        line += f"  [bold]{tr['side'].upper()}[/] {abs(tr['delta_units']):.4f} (cost ${tr['cost']:,.2f})"
    console.print(line)


def cmd_paper_run(args):
    trader = _get_trader(args)
    console.print(f"[bold cyan]Paper run:[/] {trader.name} — {trader.symbol} {trader.interval} "
                  f"{trader.strategy}{trader.params}")
    if args.once:
        _print_tick(trader.tick())
        return
    console.print(f"[dim]polling every {args.poll}s; Ctrl-C to stop (state is saved each tick)[/]")
    try:
        for status in trader.run(poll_seconds=args.poll, max_iter=args.max_iter):
            _print_tick(status)
    except KeyboardInterrupt:
        console.print("\n[yellow]stopped — state saved.[/]")


def cmd_paper_status(args):
    args._status = True
    name = args.name or f"{args.symbol}_{args.interval}_{args.strategy}"
    if not PaperTrader.exists(name):
        console.print(f"[red]no paper run named '{name}'. Start one with `paper-run`.[/]")
        return
    trader = PaperTrader.load(name)
    pf = trader.portfolio
    curve = pf.equity_curve
    last_price = curve[-1]["price"] if curve else float("nan")
    equity = pf.equity(last_price) if curve else pf.cash
    ret = (equity / pf.initial_equity - 1.0) if pf.initial_equity else float("nan")
    total_cost = sum(t["cost"] for t in pf.trades)

    t = Table(title=f"Paper run — {trader.name}", title_style="bold cyan")
    t.add_column("field", style="bold"); t.add_column("value", justify="right")
    t.add_row("symbol / interval", f"{trader.symbol} / {trader.interval}")
    t.add_row("strategy", f"{trader.strategy}{trader.params}")
    t.add_row("started", str(trader.started_at))
    t.add_row("last bar", str(trader.last_bar_time))
    t.add_row("bars processed", str(len(curve)))
    t.add_row("initial equity", f"${pf.initial_equity:,.2f}")
    t.add_row("current equity", f"${equity:,.2f}")
    t.add_row("return", _pct(ret))
    t.add_row("position (units)", f"{pf.position_units:,.6f}")
    t.add_row("position fraction", _pct(pf.position_fraction(last_price)) if curve else "—")
    t.add_row("cash", f"${pf.cash:,.2f}")
    t.add_row("# trades", str(len(pf.trades)))
    t.add_row("total costs paid", f"${total_cost:,.2f}")
    console.print(t)

    if pf.trades:
        tt = Table(title="last 5 trades", title_style="dim")
        for c in ("time", "side", "price", "Δunits", "cost", "equity_after"):
            tt.add_column(c, justify="right")
        for tr in pf.trades[-5:]:
            tt.add_row(str(tr["time"]), tr["side"], f"{tr['price']:,.4f}",
                       f"{tr['delta_units']:+.4f}", f"${tr['cost']:,.2f}",
                       f"${tr['equity_after']:,.2f}")
        console.print(tt)


def cmd_paper_reset(args):
    name = args.name or f"{args.symbol}_{args.interval}_{args.strategy}"
    path = PaperTrader._path(name)
    if path.exists():
        path.unlink()
        console.print(f"[yellow]deleted paper run '{name}'.[/]")
    else:
        console.print(f"[dim]no paper run '{name}' to delete.[/]")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edgeproof", description="Prove a trading edge net of costs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--symbol", default="BTCUSDT")
        sp.add_argument("--interval", default="1h")
        sp.add_argument("--bars", type=int, default=4000)
        sp.add_argument("--strategy", default="ma_cross", choices=list(REGISTRY))
        # ma_cross
        sp.add_argument("--fast", type=int, default=20)
        sp.add_argument("--slow", type=int, default=50)
        sp.add_argument("--allow-short", action="store_true")
        # vol_shock_drift
        sp.add_argument("--vol-window", type=int, default=96)
        sp.add_argument("--vol-z", type=float, default=3.0)
        sp.add_argument("--ret-thresh", type=float, default=0.01)
        sp.add_argument("--hold", type=int, default=6)
        sp.add_argument("--mode", default="momentum", choices=["momentum", "reversal"])
        sp.add_argument("--event-side", default="both", choices=["both", "up", "down"])
        sp.add_argument("--long-only", action="store_true")
        # shared
        sp.add_argument("--capital", type=float, default=None,
                        help="notional in USD, enables capacity checks")
        sp.add_argument("--fee-bps", type=float, default=7.5)
        sp.add_argument("--slippage-bps", type=float, default=2.0)
        sp.add_argument("--spread-bps", type=float, default=1.0)
        sp.add_argument("--refresh", action="store_true", help="ignore cache, refetch")

    b = sub.add_parser("backtest", help="run a single backtest")
    common(b); b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("scan", help="grid-search and deflate the best Sharpe")
    common(s); s.set_defaults(func=cmd_scan)

    w = sub.add_parser("walkforward", aliases=["wf"],
                       help="choose params in-sample, score out-of-sample, rolling")
    common(w)
    w.add_argument("--train-bars", type=int, default=1500)
    w.add_argument("--test-bars", type=int, default=400)
    w.add_argument("--anchored", action="store_true", help="expanding train window instead of rolling")
    w.set_defaults(func=cmd_walkforward)

    def paper_common(sp):
        sp.add_argument("--name", default=None, help="run name (default: symbol_interval_strategy)")
        sp.add_argument("--equity", type=float, default=10_000.0, help="starting fake capital")

    pr = sub.add_parser("paper-run", help="run a strategy live on fake money")
    common(pr); paper_common(pr)
    pr.add_argument("--once", action="store_true", help="single tick then exit")
    pr.add_argument("--poll", type=int, default=60, help="seconds between ticks")
    pr.add_argument("--max-iter", type=int, default=None, help="stop after N ticks")
    pr.set_defaults(func=cmd_paper_run)

    ps = sub.add_parser("paper-status", help="show a paper run's portfolio & trades")
    common(ps); paper_common(ps)
    ps.set_defaults(func=cmd_paper_status)

    px = sub.add_parser("paper-reset", help="delete a paper run's saved state")
    common(px); paper_common(px)
    px.set_defaults(func=cmd_paper_reset)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
