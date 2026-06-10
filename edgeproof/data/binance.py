"""Free OHLCV data from Binance public REST — no API key required.

Public market data (klines) needs no authentication. We paginate backwards
from the most recent bar and cache the result to parquet so repeated backtests
don't re-hit the network.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

_BASE = "https://api.binance.com/api/v3/klines"
_MAX_LIMIT = 1000  # Binance hard cap per request

# interval string -> milliseconds
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


def _cache_path(symbol: str, interval: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{symbol.upper()}_{interval}.parquet"


def _raw_request(symbol: str, interval: str, end_time: int, limit: int) -> list:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
        "endTime": end_time,
    }
    for attempt in range(4):
        try:
            r = requests.get(_BASE, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def _klines_to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base", "taker_quote", "ignore",
        ],
    )
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]]
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").drop(columns=["open_time"])
    return df


def fetch_klines(
    symbol: str,
    interval: str = "1h",
    bars: int = 2000,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of the most recent `bars` OHLCV bars.

    Index is a UTC DatetimeIndex; columns: open, high, low, close, volume,
    quote_volume (quote_volume is the per-bar traded value in USDT — used for
    capacity checks).
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported interval {interval!r}; one of {list(INTERVAL_MS)}")

    cache = _cache_path(symbol, interval)
    if cache.exists() and not force_refresh:
        cached = pd.read_parquet(cache)
        if len(cached) >= bars:
            return cached.iloc[-bars:].copy()

    frames: list[pd.DataFrame] = []
    collected = 0
    end_time = int(time.time() * 1000)
    while collected < bars:
        limit = min(_MAX_LIMIT, bars - collected)
        rows = _raw_request(symbol, interval, end_time, limit)
        if not rows:
            break
        df = _klines_to_df(rows)
        frames.append(df)
        collected += len(df)
        earliest = int(rows[0][0])
        end_time = earliest - 1  # step back before the oldest bar we just got
        if len(rows) < limit:
            break  # ran out of history
        time.sleep(0.25)  # be polite to the public endpoint

    if not frames:
        raise RuntimeError(f"no data returned for {symbol} {interval}")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.to_parquet(cache)
    return out.iloc[-bars:].copy()
