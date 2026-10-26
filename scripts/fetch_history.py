#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generic history fetch: TMF futures (TMFR1) or stocks → data/cache/{SYMBOL}/."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.core.config import notify_prefix
from src.data.history_pipeline import (
    daterange,
    fetch_and_cache_tmf_1m_day,
    fetch_and_cache_tmf_day,
    fetch_stock_daily_finmind,
    fetch_stock_day,
    is_futures,
    probe_tmfr1_earliest,
    resolve_preset,
    weekdays_only,
    write_report,
)
from src.data.local_store import LocalStore

PROGRESS_EVERY = 25
# Shioaji SIM historically returns empty under burst load; pace day fetches.
DAY_PAUSE_SEC = 0.6
EMPTY_RETRY_PAUSE_SEC = 1.5


def main() -> int:
    p = argparse.ArgumentParser(
        description="tidemark fetch history (TMF via TMFR1, or stocks) → local parquet"
    )
    p.add_argument(
        "--symbol",
        default="TMF",
        help="TMF (default) or stock code e.g. 2330",
    )
    p.add_argument("--date", help="single day YYYY-MM-DD")
    p.add_argument("--start", help="range start YYYY-MM-DD")
    p.add_argument("--end", help="range end YYYY-MM-DD")
    p.add_argument(
        "--preset",
        choices=["recent6m", "full"],
        help="recent6m ≈ last 6 months; full = listing→yesterday (TMF) or ~2y (stocks)",
    )
    p.add_argument(
        "--source",
        default="shioaji",
        choices=["shioaji", "finmind"],
        help="default shioaji. finmind: futures daily / stock daily; tick needs Sponsor",
    )
    p.add_argument(
        "--kind",
        default="1m",
        choices=["1s", "tick", "1m", "daily"],
        help="default 1m (small/fast via kbars); 1s from ticks; tick needs --keep-tick to persist",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--keep-tick",
        action="store_true",
        help="also save raw tick parquet (large); default off",
    )
    p.add_argument(
        "--include-weekend",
        action="store_true",
        help="also request Sat/Sun (usually empty)",
    )
    p.add_argument(
        "--probe-earliest",
        action="store_true",
        help="binary-search earliest TMFR1 day (TMF only); write report and exit",
    )
    p.add_argument(
        "--report",
        default="outputs/fetch_history_report.json",
        help="result JSON path (agent reads this; do not poll logs)",
    )
    args = p.parse_args()

    symbol = args.symbol.upper()
    futures = is_futures(symbol)
    report_path = ROOT / args.report
    t0 = time.time()

    if args.probe_earliest:
        if symbol != "TMF":
            p.error("--probe-earliest only supported for --symbol TMF")
            return 2
        probe = probe_tmfr1_earliest()
        payload = {
            "status": "DONE",
            "mode": "probe_earliest",
            "symbol": "TMF",
            "continuous_code": "TMFR1",
            "earliest": probe.get("earliest"),
            "elapsed_sec": round(time.time() - t0, 1),
            **probe,
        }
        write_report(payload, report_path)
        # also keep a dedicated probe snapshot
        write_report(payload, ROOT / "outputs" / "tmfr1_probe.json")
        return 0 if probe.get("earliest") else 1

    if args.preset:
        start, end = resolve_preset(symbol, args.preset)
        dates = daterange(start, end)
        logger.info(f"{notify_prefix()} preset={args.preset} {symbol} range={start}..{end}")
    elif args.date:
        dates = [args.date]
        start = end = args.date
    elif args.start and args.end:
        dates = daterange(args.start, args.end)
        start, end = args.start, args.end
    else:
        p.error("provide --preset, --date, --probe-earliest, or both --start and --end")
        return 2

    if not args.include_weekend and not args.date:
        dates = weekdays_only(dates)

    store = LocalStore()
    results: list[dict] = []
    ok = 0
    miss = 0

    write_report(
        {
            "status": "RUNNING",
            "symbol": symbol,
            "asset": "futures" if futures else "stock",
            "kind": args.kind,
            "source": args.source,
            "start": start,
            "end": end,
            "requested_days": len(dates),
            "ok": 0,
            "miss": 0,
            "continuous_code": "TMFR1" if futures and symbol == "TMF" else None,
            "elapsed_sec": 0,
        },
        report_path,
    )

    # ---- daily path (range once) ----
    if args.kind == "daily":
        if futures and symbol == "TMF":
            from src.data.tmf_pipeline import fetch_and_cache_tmf_daily

            df = fetch_and_cache_tmf_daily(start, end, store=store)
            rows = 0 if df is None else len(df)
            payload = {
                "status": "DONE",
                "symbol": symbol,
                "kind": "daily",
                "source": "finmind",
                "start": start,
                "end": end,
                "rows": rows,
                "ok": rows > 0,
                "elapsed_sec": round(time.time() - t0, 1),
                "note": "TMFR1 used for tick/1s/1m path; daily via FinMind",
            }
            write_report(payload, report_path)
            logger.info(f"{notify_prefix()} daily rows={rows}")
            return 0 if rows > 0 else 1

        summary = fetch_stock_daily_finmind(symbol, start, end, store)
        payload = {
            "status": "DONE",
            "symbol": symbol,
            "kind": "daily",
            "source": "finmind",
            "start": start,
            "end": end,
            **summary,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        write_report(payload, report_path)
        return 0 if summary.get("rows", 0) > 0 else 1

    # ---- day-by-day ----
    shioaji_f = None
    shioaji_s = None
    cache_kind = "1m"
    try:
        if futures:
            if args.source == "shioaji":
                from src.data.shioaji_futures import ShioajiFutures

                shioaji_f = ShioajiFutures(simulation=True)
            total = len(dates)
            for i, d in enumerate(dates, start=1):
                if args.kind == "1m":
                    summary = fetch_and_cache_tmf_1m_day(
                        d,
                        store=store,
                        force=args.force,
                        shioaji=shioaji_f,
                        symbol=symbol,
                    )
                    cache_kind = "1m"
                else:
                    # 1s or tick → tick fetch → 1s (+ optional tick persist)
                    summary = fetch_and_cache_tmf_day(
                        d,
                        store=store,
                        force=args.force,
                        source=args.source,
                        shioaji=shioaji_f,
                        symbol=symbol,
                        keep_tick=args.keep_tick or args.kind == "tick",
                        also_1m=True,
                    )
                    cache_kind = "1s"
                # one retry on empty (rate-limit false miss)
                hit_now = (
                    summary.get("sec_rows", 0) > 0
                    or summary.get("rows", 0) > 0
                    or summary.get("cached")
                )
                if not hit_now and not summary.get("cached"):
                    time.sleep(EMPTY_RETRY_PAUSE_SEC)
                    if args.kind == "1m":
                        summary = fetch_and_cache_tmf_1m_day(
                            d,
                            store=store,
                            force=True,
                            shioaji=shioaji_f,
                            symbol=symbol,
                        )
                    else:
                        summary = fetch_and_cache_tmf_day(
                            d,
                            store=store,
                            force=True,
                            source=args.source,
                            shioaji=shioaji_f,
                            symbol=symbol,
                            keep_tick=args.keep_tick or args.kind == "tick",
                            also_1m=True,
                        )
                summary["symbol"] = symbol
                results.append(summary)
                time.sleep(DAY_PAUSE_SEC)
                tag = f"[{i}/{total}]"
                hit = (
                    summary.get("sec_rows", 0) > 0
                    or summary.get("rows", 0) > 0
                    or summary.get("cached")
                )
                if hit:
                    ok += 1
                    logger.info(f"{notify_prefix()} {tag} ok {summary}")
                else:
                    miss += 1
                    logger.warning(f"{notify_prefix()} {tag} miss {summary}")
                if i % PROGRESS_EVERY == 0 or i == total:
                    cached_so_far = store.list_dates(symbol, cache_kind)
                    write_report(
                        {
                            "status": "RUNNING",
                            "symbol": symbol,
                            "asset": "futures",
                            "kind": args.kind,
                            "cache_kind": cache_kind,
                            "source": args.source,
                            "start": start,
                            "end": end,
                            "requested_days": total,
                            "progress_i": i,
                            "ok": ok,
                            "miss": miss,
                            "cached_dates": len(cached_so_far),
                            "first_cached": cached_so_far[0] if cached_so_far else None,
                            "last_cached": cached_so_far[-1] if cached_so_far else None,
                            "continuous_code": "TMFR1" if symbol == "TMF" else None,
                            "elapsed_sec": round(time.time() - t0, 1),
                        },
                        report_path,
                    )
        else:
            from src.data.shioaji_stocks import ShioajiStocks

            shioaji_s = ShioajiStocks(simulation=True)
            stock_kind = "tick" if args.kind == "tick" else "1m"
            cache_kind = "1m"
            total = len(dates)
            for i, d in enumerate(dates, start=1):
                summary = fetch_stock_day(
                    symbol,
                    d,
                    store=store,
                    shioaji=shioaji_s,
                    force=args.force,
                    kind=stock_kind,
                    keep_tick=args.keep_tick or args.kind == "tick",
                )
                results.append(summary)
                tag = f"[{i}/{total}]"
                if summary.get("rows", 0) > 0 or summary.get("cached"):
                    ok += 1
                    logger.info(f"{notify_prefix()} {tag} ok {summary}")
                else:
                    miss += 1
                    logger.warning(f"{notify_prefix()} {tag} miss {summary}")
                if i % PROGRESS_EVERY == 0 or i == total:
                    cached_so_far = store.list_dates(symbol, cache_kind)
                    write_report(
                        {
                            "status": "RUNNING",
                            "symbol": symbol,
                            "asset": "stock",
                            "kind": args.kind,
                            "cache_kind": cache_kind,
                            "source": args.source,
                            "start": start,
                            "end": end,
                            "requested_days": total,
                            "progress_i": i,
                            "ok": ok,
                            "miss": miss,
                            "cached_dates": len(cached_so_far),
                            "first_cached": cached_so_far[0] if cached_so_far else None,
                            "last_cached": cached_so_far[-1] if cached_so_far else None,
                            "elapsed_sec": round(time.time() - t0, 1),
                        },
                        report_path,
                    )
    finally:
        if shioaji_f is not None:
            shioaji_f.close()
        if shioaji_s is not None:
            shioaji_s.close()

    cached_dates = store.list_dates(symbol, cache_kind)
    payload = {
        "status": "DONE",
        "symbol": symbol,
        "asset": "futures" if futures else "stock",
        "kind": args.kind,
        "cache_kind": cache_kind,
        "source": args.source,
        "keep_tick": bool(args.keep_tick or args.kind == "tick"),
        "start": start,
        "end": end,
        "requested_days": len(dates),
        "ok": ok,
        "miss": miss,
        "cached_dates": len(cached_dates),
        "first_cached": cached_dates[0] if cached_dates else None,
        "last_cached": cached_dates[-1] if cached_dates else None,
        "earliest": cached_dates[0] if cached_dates else None,
        "continuous_code": "TMFR1" if futures and symbol == "TMF" else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "report_note": "TMFR1 = 微型台指近月連續契約；舊日期請用連續碼勿用當月碼",
        "days": results[-20:],  # tail only to keep report small
    }
    write_report(payload, report_path)
    logger.info(
        f"{notify_prefix()} done {symbol} ok={ok} miss={miss} total={len(dates)} "
        f"kind={args.kind} source={args.source}"
    )
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
