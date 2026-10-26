from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.local_store import LocalStore
from src.data.tmf_pipeline import daterange


def load_1s_range(
    start: str,
    end: str,
    *,
    store: LocalStore | None = None,
    symbol: str = "TMF",
) -> pd.DataFrame:
    store = store or LocalStore()
    frames: list[pd.DataFrame] = []
    for d in daterange(start, end):
        df = store.load(symbol, d, "1s")
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    return out.sort_values("ts").reset_index(drop=True)


def to_1m(bars_1s: pd.DataFrame) -> pd.DataFrame:
    """Resample 1s OHLCV → 1m for fast strategy search (same session label = majority)."""
    if bars_1s is None or bars_1s.empty:
        return pd.DataFrame()
    df = bars_1s.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    g = df.resample("1min").agg(agg).dropna(subset=["close"])
    if "session" in df.columns:
        # take last known session in the minute
        sess = df["session"].resample("1min").last()
        g["session"] = sess
    else:
        from src.data.sessions import label_session

        g["session"] = g.index.map(label_session)
    g = g[g["session"].isin(["day", "night"])].reset_index()
    return g


def cache_1m_from_1s(
    start: str,
    end: str,
    *,
    store: LocalStore | None = None,
    symbol: str = "TMF",
    force: bool = False,
) -> int:
    """Materialize per-day 1m parquet beside 1s for reuse."""
    store = store or LocalStore()
    n = 0
    for d in daterange(start, end):
        path = store.path_for(symbol, d, "1m")
        if path.exists() and not force:
            existing = store.load(symbol, d, "1m")
            if existing is not None and not existing.empty:
                n += 1
                continue
        s1 = store.load(symbol, d, "1s")
        if s1 is None or s1.empty:
            continue
        m1 = to_1m(s1)
        if m1.empty:
            continue
        store.save(m1, symbol, d, "1m")
        n += 1
    return n


def load_1m_range(start: str, end: str, *, store: LocalStore | None = None, symbol: str = "TMF") -> pd.DataFrame:
    store = store or LocalStore()
    frames: list[pd.DataFrame] = []
    for d in daterange(start, end):
        df = store.load(symbol, d, "1m")
        if df is None or df.empty:
            # lazy build from 1s if present
            s1 = store.load(symbol, d, "1s")
            if s1 is None or s1.empty:
                continue
            df = to_1m(s1)
            if not df.empty:
                store.save(df, symbol, d, "1m")
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    return out.sort_values("ts").reset_index(drop=True)
