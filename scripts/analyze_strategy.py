#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detailed analysis for a single strategy combo.
Writes outputs/*_report.json — agents should read that file, not logs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.engine import SecondBarBacktester
from src.backtest.metrics import summarize
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range
from src.signals.candidates import build_signal
from src.strategy.tmf_session import TmfSessionStrategy


def _backtest(bars, signal, trading, slip):
    strat = TmfSessionStrategy(
        signal=signal,
        max_lots=trading["max_lots"],
        capital=trading["capital"],
    )
    bt = SecondBarBacktester(
        strat,
        capital=trading["capital"],
        point_value=trading["point_value"],
        fee_per_side=trading.get("fee_per_side", 20),
        tax_rate=trading.get("tax_rate", 0.00002),
        slip_ticks=slip,
    )
    return bt.run(bars)


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark detailed strategy analysis")
    p.add_argument("--signal", default="breakout")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--sessions", default="day", help="comma list: day,night")
    p.add_argument("--is-start", default="2026-07-16")
    p.add_argument("--is-end", default="2026-09-15")
    p.add_argument("--oos-start", default="2026-09-16")
    p.add_argument("--oos-end", default="2026-10-21")
    p.add_argument("--tag", default="breakout40_day")
    args = p.parse_args()

    settings = load_settings()
    trading = settings["trading"]
    slip = settings.get("backtest", {}).get("slip_ticks", 1)
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    params = {"lookback": args.lookback, "sessions": sessions}
    if args.signal == "ema_mom":
        params = {"ema_period": args.lookback, "sessions": sessions}
    elif args.signal == "rsi_mr":
        params = {
            "rsi_period": 6,
            "rsi_low": 25,
            "rsi_high": 75,
            "sessions": sessions,
        }

    ranges = {
        "IS": (args.is_start, args.is_end),
        "OOS": (args.oos_start, args.oos_end),
        "FULL": (args.is_start, args.oos_end),
    }

    sections = {}
    for label, (start, end) in ranges.items():
        bars = load_1m_range(start, end)
        if bars.empty:
            sections[label] = {"error": "no bars", "range": [start, end]}
            continue
        # Keep night bars in history for indicators; SessionFilter limits entries.
        signal = build_signal(args.signal, params)
        result = _backtest(bars, signal, trading, slip)
        sections[label] = summarize(result, capital=trading["capital"], label=label)
        sections[label]["range"] = [start, end]
        sections[label]["n_bars"] = len(bars)

    verdict_bits = []
    is_pnl = sections.get("IS", {}).get("net_pnl")
    oos_pnl = sections.get("OOS", {}).get("net_pnl")
    if isinstance(is_pnl, (int, float)) and isinstance(oos_pnl, (int, float)):
        if is_pnl > 0 and oos_pnl > 0:
            verdict_bits.append("IS and OOS both positive")
        elif oos_pnl > 0:
            verdict_bits.append("OOS positive but check IS weakness")
        else:
            verdict_bits.append("OOS not profitable — do not promote")

    report = {
        "status": "DONE",
        "brand": "tidemark",
        "tag": args.tag,
        "signal": args.signal,
        "params": params,
        "point_value": trading["point_value"],
        "max_lots": trading["max_lots"],
        "capital": trading["capital"],
        "sections": sections,
        "verdict": "; ".join(verdict_bits) if verdict_bits else "n/a",
    }

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{args.tag}_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(f"{notify_prefix()} wrote {path}")
    print(json.dumps({"status": "DONE", "path": str(path), "verdict": report["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
