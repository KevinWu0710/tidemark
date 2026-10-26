#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replay local 1s cache through SecondBarBacktester (no look-ahead)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.engine import SecondBarBacktester
from src.core.config import load_settings, notify_prefix
from src.data.local_store import LocalStore
from src.data.tmf_pipeline import daterange
from src.strategy.tmf_session import TmfSessionStrategy


def load_bars(store: LocalStore, start: str, end: str, symbol: str = "TMF") -> pd.DataFrame:
    frames = []
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


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark second-bar backtest from cache")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()

    settings = load_settings()
    trading = settings["trading"]
    store = LocalStore(settings)
    bars = load_bars(store, args.start, args.end)
    if bars.empty:
        logger.error(f"{notify_prefix(settings)} no 1s cache in range — run fetch_tmf.py first")
        return 1

    strat_cfg = next(
        (s for s in settings.get("strategies", []) if s.get("id") == "tmf_v1"),
        {},
    )
    params = dict(strat_cfg.get("params") or {})
    strat = TmfSessionStrategy(
        max_lots=trading["max_lots"],
        capital=trading["capital"],
        params=params,
    )
    bt = SecondBarBacktester(
        strat,
        capital=trading["capital"],
        point_value=trading["point_value"],
        fee_per_side=trading.get("fee_per_side", 20),
        tax_rate=trading.get("tax_rate", 0.00002),
        slip_ticks=settings.get("backtest", {}).get("slip_ticks", 1),
    )
    result = bt.run(bars)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "brand": "tidemark",
        "start": args.start,
        "end": args.end,
        "n_steps": result.n_steps,
        "n_trades": len(result.trades),
        "final_equity": result.final_equity,
        "capital": trading["capital"],
    }
    path = out_dir / f"backtest_{args.start}_{args.end}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"{notify_prefix(settings)} {summary}")
    logger.info(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
