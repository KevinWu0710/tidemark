from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from src.core.config import ROOT, notify_prefix
from src.data.local_store import LocalStore
from src.data.tmf_pipeline import (
    daterange,
    fetch_and_cache_tmf_1m_day,
    fetch_and_cache_tmf_day,
    ticks_to_1s,
)

TMF_LISTING = date(2024, 7, 29)
FUTURES_SYMBOLS = frozenset({"TMF", "MTX", "TX", "TXF", "MXF"})


def yesterday() -> date:
    return date.today() - timedelta(days=1)


def weekdays_only(dates: list[str]) -> list[str]:
    out = []
    for d in dates:
        if date.fromisoformat(d).weekday() < 5:
            out.append(d)
    return out


def resolve_preset(symbol: str, name: str) -> tuple[str, str]:
    end = yesterday()
    sym = symbol.upper()
    if name == "recent6m":
        start = end - timedelta(days=183)
        if sym == "TMF" and start < TMF_LISTING:
            start = TMF_LISTING
        return start.isoformat(), end.isoformat()
    if name == "full":
        if sym == "TMF":
            return TMF_LISTING.isoformat(), end.isoformat()
        start = end - timedelta(days=730)
        return start.isoformat(), end.isoformat()
    raise ValueError(f"unknown preset: {name}")


def is_futures(symbol: str) -> bool:
    return symbol.upper() in FUTURES_SYMBOLS or symbol.upper().startswith("TMF")


def _day_has_tmfr1(
    shioaji,
    day: str,
    *,
    prefer: str = "ticks",
    retries: int = 2,
    pause_sec: float = 0.8,
) -> tuple[bool, int]:
    """Return (has_data, row_count). Retries empty results (SIM often rate-limits)."""
    n = 0
    for attempt in range(retries + 1):
        if prefer == "ticks":
            df = shioaji.ticks(day, contract_code="TMFR1")
        else:
            df = shioaji.kbars(day, contract_code="TMFR1")
        n = 0 if df is None else len(df)
        if n > 0:
            return True, n
        if attempt < retries:
            time.sleep(pause_sec * (attempt + 1))
    return False, n


def probe_tmfr1_earliest(
    *,
    lo: Optional[str] = None,
    hi: Optional[str] = None,
    shioaji=None,
    prefer: str = "ticks",
) -> dict:
    """
    Find earliest weekday with non-empty TMFR1 ticks.

    Strategy: check listing day first (avoids binary-search false negatives
    under Shioaji SIM rate limits). If empty, walk forward weekly with pauses.
    """
    from src.data.shioaji_futures import ShioajiFutures

    close_after = shioaji is None
    shioaji = shioaji or ShioajiFutures(simulation=True)
    start = date.fromisoformat(lo or TMF_LISTING.isoformat())
    end = date.fromisoformat(hi or yesterday().isoformat())
    days = weekdays_only(daterange(start.isoformat(), end.isoformat()))
    probed: list[dict] = []

    def check(d: str) -> bool:
        ok, n = _day_has_tmfr1(shioaji, d, prefer=prefer)
        probed.append({"date": d, "ok": ok, "rows": n, "via": prefer})
        logger.info(f"{notify_prefix()} probe {d} via={prefer} rows={n}")
        time.sleep(0.5)
        return ok

    try:
        if not days:
            return {
                "earliest": None,
                "lo": start.isoformat(),
                "hi": end.isoformat(),
                "error": "empty_range",
                "probed": probed,
            }
        # Fast path: listing day itself (verified TMFR1 has ticks from 2024-07-29)
        if check(days[0]):
            return {
                "earliest": days[0],
                "tick_rows": probed[-1]["rows"],
                "lo": start.isoformat(),
                "hi": end.isoformat(),
                "probed_n": len(probed),
                "probed": probed,
                "continuous_code": "TMFR1",
                "prefer": prefer,
                "method": "listing_day",
                "note": "TMFR1 = 微型台指近月連續契約；上市日即有 tick",
            }
        # Walk forward (weekly samples then fill) to avoid rate-limit false negatives
        step = 5  # every ~week of weekdays
        earliest = None
        for i in range(0, len(days), step):
            if check(days[i]):
                # walk back within the step window for true earliest
                window = days[max(0, i - step + 1) : i + 1]
                earliest = days[i]
                for d in window:
                    if check(d):
                        earliest = d
                        break
                break
        if earliest is None:
            return {
                "earliest": None,
                "lo": start.isoformat(),
                "hi": end.isoformat(),
                "error": "no_data_in_range",
                "probed": probed[-40:],
                "continuous_code": "TMFR1",
            }
        tick_rows = next(
            (p["rows"] for p in reversed(probed) if p["date"] == earliest and p["ok"]),
            None,
        )
        return {
            "earliest": earliest,
            "tick_rows": tick_rows,
            "lo": start.isoformat(),
            "hi": end.isoformat(),
            "probed_n": len(probed),
            "probed": probed[-40:],
            "continuous_code": "TMFR1",
            "prefer": prefer,
            "method": "forward_walk",
            "note": "TMFR1 = 微型台指近月連續契約",
        }
    finally:
        if close_after:
            shioaji.close()


def fetch_stock_day(
    symbol: str,
    day: str,
    *,
    store: LocalStore,
    shioaji,
    force: bool = False,
    kind: str = "1m",
    keep_tick: bool = False,
) -> dict:
    symbol = symbol.upper()
    target = "tick" if kind == "tick" else "1m"
    if not force and store.has(symbol, day, target):
        existing = store.load(symbol, day, target)
        return {
            "symbol": symbol,
            "date": day,
            "cached": True,
            "rows": len(existing),
            "kind": target,
        }

    if kind == "tick":
        raw = shioaji.ticks(symbol, day)
        if raw is None or raw.empty:
            return {"symbol": symbol, "date": day, "cached": False, "rows": 0, "error": "empty"}
        if keep_tick:
            store.save(raw, symbol, day, "tick")
        sec = ticks_to_1s(raw)
        if not sec.empty:
            store.save(sec, symbol, day, "1s")
            from src.data.bars import to_1m

            m1 = to_1m(sec)
            if not m1.empty:
                store.save(m1, symbol, day, "1m")
        return {
            "symbol": symbol,
            "date": day,
            "cached": False,
            "rows": len(raw),
            "kind": "tick",
            "kept_tick": keep_tick,
        }

    kb = shioaji.kbars(symbol, day)
    if kb is None or kb.empty:
        return {"symbol": symbol, "date": day, "cached": False, "rows": 0, "error": "empty"}
    from src.data.sessions import label_session

    if "session" not in kb.columns:
        kb = kb.copy()
        kb["session"] = kb["ts"].map(label_session)
    store.save(kb, symbol, day, "1m")
    return {"symbol": symbol, "date": day, "cached": False, "rows": len(kb), "kind": "1m"}


def fetch_stock_daily_finmind(symbol: str, start: str, end: str, store: LocalStore) -> dict:
    from src.data.finmind_futures import FinMindFutures

    api = FinMindFutures()
    df = api._get(
        {
            "dataset": "TaiwanStockPrice",
            "data_id": symbol.upper(),
            "start_date": start,
            "end_date": end,
        }
    )
    if df is None or df.empty:
        return {"symbol": symbol, "rows": 0, "error": "empty", "kind": "daily"}
    path = store.cache_dir / symbol.upper() / "daily" / f"{start}_{end}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return {"symbol": symbol, "rows": len(df), "path": str(path), "kind": "daily"}


def write_report(payload: dict, path: Optional[Path] = None) -> Path:
    out = path or (ROOT / "outputs" / "fetch_history_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"{notify_prefix()} wrote {out}")
    return out


# re-export for CLI convenience
__all__ = [
    "TMF_LISTING",
    "daterange",
    "fetch_and_cache_tmf_1m_day",
    "fetch_and_cache_tmf_day",
    "fetch_stock_daily_finmind",
    "fetch_stock_day",
    "is_futures",
    "probe_tmfr1_earliest",
    "resolve_preset",
    "weekdays_only",
    "write_report",
    "yesterday",
]
