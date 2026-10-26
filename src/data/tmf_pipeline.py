from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from src.data.finmind_futures import FinMindFutures
from src.data.local_store import LocalStore
from src.data.sessions import label_session
from src.data.shioaji_futures import ShioajiFutures


def _pick_front_month(tick: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the single contract_date with max volume that day (front / most liquid).
    Drop spread rows (contract_date containing '/').
    """
    if tick.empty or "contract_date" not in tick.columns:
        return tick
    df = tick.copy()
    df["contract_date"] = df["contract_date"].astype(str)
    df = df[~df["contract_date"].str.contains("/", na=False)]
    if df.empty:
        return df
    vol = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df = df.assign(_vol=vol)
    ranked = df.groupby("contract_date")["_vol"].sum().sort_values(ascending=False)
    front = str(ranked.index[0])
    out = df[df["contract_date"] == front].drop(columns=["_vol"])
    logger.info(f"front month={front} rows={len(out)} / {len(tick)}")
    return out


def ticks_to_1s(tick: pd.DataFrame) -> pd.DataFrame:
    """Resample trade ticks to 1-second OHLCV; label session."""
    if tick.empty:
        return pd.DataFrame()

    df = tick.copy()
    if "date" not in df.columns or "price" not in df.columns:
        raise ValueError("tick missing date/price columns")

    df["ts"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["ts", "price"]).sort_values("ts")
    if df.empty:
        return pd.DataFrame()

    g = (
        df.set_index("ts")
        .resample("1s")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    if "contract_date" in df.columns:
        # attach dominant contract for the day
        g["contract_date"] = str(df["contract_date"].iloc[0])
    g["session"] = g["ts"].map(label_session)
    g = g[g["session"].isin(["day", "night"])].reset_index(drop=True)
    return g


def _fetch_raw_ticks(
    date: str,
    *,
    source: str,
    symbol: str,
    finmind: Optional[FinMindFutures],
    shioaji: Optional[ShioajiFutures],
) -> pd.DataFrame:
    if source == "shioaji":
        assert shioaji is not None
        return shioaji.ticks(date)
    assert finmind is not None
    return finmind.futures_tick(symbol, date)


def fetch_and_cache_tmf_1m_day(
    date: str,
    *,
    store: Optional[LocalStore] = None,
    shioaji: Optional[ShioajiFutures] = None,
    force: bool = False,
    symbol: str = "TMF",
) -> dict:
    """
    Fast small path: Shioaji kbars (TMFR1) → 1m parquet only.
    Prefer this for signal research / long backfill.
    """
    store = store or LocalStore()
    if not force and store.has(symbol, date, "1m"):
        existing = store.load(symbol, date, "1m")
        return {
            "date": date,
            "cached": True,
            "source": "shioaji",
            "kind": "1m",
            "rows": len(existing),
            "sec_rows": len(existing),  # CLI ok-counter compatibility
            "path_1m": str(store.path_for(symbol, date, "1m")),
        }

    shioaji = shioaji or ShioajiFutures(simulation=True)
    via = "kbars"
    kb = shioaji.kbars(date)
    df = pd.DataFrame() if kb is None else kb.copy()
    if not df.empty:
        if "session" not in df.columns:
            df["session"] = df["ts"].map(label_session)
        df = df[df["session"].isin(["day", "night"])].reset_index(drop=True)

    # Some days return empty kbars while ticks exist — fall back to tick→1s→1m
    if df.empty:
        via = "ticks"
        raw = shioaji.ticks(date)
        if raw is None or raw.empty:
            return {
                "date": date,
                "cached": False,
                "source": "shioaji",
                "kind": "1m",
                "rows": 0,
                "sec_rows": 0,
                "error": "empty",
                "via": via,
            }
        sec = ticks_to_1s(raw)
        if sec.empty:
            return {
                "date": date,
                "cached": False,
                "source": "shioaji",
                "kind": "1m",
                "rows": 0,
                "sec_rows": 0,
                "error": "resample_empty",
                "via": via,
            }
        # also land 1s so later 1s backtests can reuse
        if not store.has(symbol, date, "1s"):
            store.save(sec, symbol, date, "1s")
        from src.data.bars import to_1m

        df = to_1m(sec)

    if df.empty:
        return {
            "date": date,
            "cached": False,
            "source": "shioaji",
            "kind": "1m",
            "rows": 0,
            "sec_rows": 0,
            "error": "session_empty",
            "via": via,
        }
    store.save(df, symbol, date, "1m")
    return {
        "date": date,
        "cached": False,
        "source": "shioaji",
        "kind": "1m",
        "rows": len(df),
        "sec_rows": len(df),
        "via": via,
        "path_1m": str(store.path_for(symbol, date, "1m")),
        "sessions": sorted(df["session"].unique().tolist()),
    }


def fetch_and_cache_tmf_day(
    date: str,
    *,
    store: Optional[LocalStore] = None,
    client: Optional[FinMindFutures] = None,
    shioaji: Optional[ShioajiFutures] = None,
    source: str = "shioaji",
    force: bool = False,
    symbol: str = "TMF",
    keep_tick: bool = False,
    also_1m: bool = True,
) -> dict:
    """
    Fetch one day TMF ticks → 1s (+ optional 1m).
    keep_tick=False（預設）不存原始 tick，省磁碟；需要稽核再開。
    """
    store = store or LocalStore()
    source = (source or "shioaji").lower()

    tick_path = store.path_for(symbol, date, "tick")
    sec_path = store.path_for(symbol, date, "1s")

    if not force and store.has(symbol, date, "1s"):
        existing = store.load(symbol, date, "1s")
        logger.info(f"cache hit 1s {date} rows={len(existing)}")
        return {
            "date": date,
            "cached": True,
            "source": source,
            "tick_rows": None,
            "sec_rows": len(existing),
            "path_1s": str(sec_path),
        }

    if source == "finmind":
        client = client or FinMindFutures()
    else:
        shioaji = shioaji or ShioajiFutures(simulation=True)

    raw = _fetch_raw_ticks(
        date, source=source, symbol=symbol, finmind=client, shioaji=shioaji
    )
    if raw is None or raw.empty:
        logger.warning(f"empty tick for {symbol} {date} source={source} — treat as miss")
        if tick_path.exists():
            tick_path.unlink()
        if sec_path.exists():
            sec_path.unlink()
        return {
            "date": date,
            "cached": False,
            "source": source,
            "tick_rows": 0,
            "sec_rows": 0,
            "error": "empty",
        }

    if keep_tick:
        store.save(raw, symbol, date, "tick")
    front = raw if source == "shioaji" else _pick_front_month(raw)
    sec = ticks_to_1s(front)
    if sec.empty:
        logger.warning(f"1s empty after resample {date}")
        return {
            "date": date,
            "cached": False,
            "source": source,
            "tick_rows": len(raw),
            "sec_rows": 0,
            "error": "resample_empty",
        }

    store.save(sec, symbol, date, "1s")
    if also_1m:
        from src.data.bars import to_1m

        m1 = to_1m(sec)
        if not m1.empty:
            store.save(m1, symbol, date, "1m")
    return {
        "date": date,
        "cached": False,
        "source": source,
        "tick_rows": len(raw),
        "sec_rows": len(sec),
        "kept_tick": keep_tick,
        "path_1s": str(sec_path),
        "sessions": sorted(sec["session"].unique().tolist()),
    }


def fetch_and_cache_tmf_daily(
    start: str,
    end: str,
    *,
    store: Optional[LocalStore] = None,
    client: Optional[FinMindFutures] = None,
    symbol: str = "TMF",
) -> pd.DataFrame:
    """FinMind TaiwanFuturesDaily → data/cache/tmf/TMF/daily/{start}_{end}.parquet"""
    store = store or LocalStore()
    client = client or FinMindFutures()
    df = client.futures_daily(symbol, start, end)
    if df is None or df.empty:
        return pd.DataFrame()
    path = store.cache_dir / symbol.upper() / "daily" / f"{start}_{end}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"saved daily {len(df)} -> {path}")
    return df


def daterange(start: str, end: str) -> list[str]:
    days = pd.date_range(start=start, end=end, freq="D")
    return [d.strftime("%Y-%m-%d") for d in days]
