#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IS/OOS signal combo scan — SLOW SecondBarBacktester path.

全歷史網格請用 scripts/scan_signals_fast.py（並行＋shard＋fast_breakout）。
本腳本僅供少數組合／秒級複核。
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
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range
from src.signals.candidates import build_signal
from src.strategy.tmf_session import TmfSessionStrategy


def _run(bars, signal, trading, slip) -> dict:
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
    result = bt.run(bars)
    pnl = result.final_equity - trading["capital"]
    return {
        "final_equity": result.final_equity,
        "pnl": pnl,
        "n_trades": len(result.trades),
        "n_steps": result.n_steps,
    }


def combo_grid():
    combos = []
    for sessions in (["day"], ["night"], ["day", "night"]):
        for ema in (10, 20, 60):
            combos.append(
                {"name": "ema_mom", "params": {"ema_period": ema, "sessions": sessions}}
            )
        for low, high in ((20, 80), (25, 75), (30, 70)):
            combos.append(
                {
                    "name": "rsi_mr",
                    "params": {
                        "rsi_period": 6,
                        "rsi_low": low,
                        "rsi_high": high,
                        "sessions": sessions,
                    },
                }
            )
        for lb in (10, 20, 40):
            combos.append(
                {"name": "breakout", "params": {"lookback": lb, "sessions": sessions}}
            )
    return combos


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark IS/OOS signal scan")
    # Defaults: full TMFR1 1m span (DATA.md IS/OOS); override if cache thinner
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument("--min-oos-trades", type=int, default=5)
    p.add_argument(
        "--require-is-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PASS needs IS pnl > 0 as well as OOS (default: on)",
    )
    args = p.parse_args()

    settings = load_settings()
    trading = settings["trading"]
    slip = settings.get("backtest", {}).get("slip_ticks", 1)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "signal_scan_report.json"
    # Agent 讀 status；RUNNING 時勿當最終結果
    report_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "brand": "tidemark",
                "is_range": [args.is_start, args.is_end],
                "oos_range": [args.oos_start, args.oos_end],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(f"{notify_prefix()} load IS {args.is_start}..{args.is_end}")
    is_bars = load_1m_range(args.is_start, args.is_end)
    logger.info(f"{notify_prefix()} load OOS {args.oos_start}..{args.oos_end}")
    oos_bars = load_1m_range(args.oos_start, args.oos_end)
    if is_bars.empty or oos_bars.empty:
        logger.error(
            f"{notify_prefix()} missing bars IS={len(is_bars)} OOS={len(oos_bars)}"
        )
        return 1

    grid = combo_grid()
    logger.info(
        f"{notify_prefix()} bars IS={len(is_bars)} OOS={len(oos_bars)} combos={len(grid)}"
    )

    rows = []
    for spec in grid:
        signal = build_signal(spec["name"], spec["params"])
        is_res = _run(is_bars, signal, trading, slip)
        signal_oos = build_signal(spec["name"], spec["params"])
        oos_res = _run(oos_bars, signal_oos, trading, slip)
        is_ok = (not args.require_is_positive) or (is_res["pnl"] > 0)
        row = {
            "signal": spec["name"],
            "params": spec["params"],
            "is_pnl": round(is_res["pnl"], 2),
            "is_trades": is_res["n_trades"],
            "oos_pnl": round(oos_res["pnl"], 2),
            "oos_trades": oos_res["n_trades"],
            "pass": bool(
                is_ok
                and oos_res["pnl"] > 0
                and oos_res["n_trades"] >= args.min_oos_trades
            ),
        }
        rows.append(row)
        flag = "PASS" if row["pass"] else "fail"
        logger.info(
            f"{notify_prefix()} [{flag}] {spec['name']} {spec['params']} "
            f"IS={row['is_pnl']} OOS={row['oos_pnl']} trades_oos={row['oos_trades']}"
        )

    rows.sort(key=lambda r: (r["pass"], r["oos_pnl"]), reverse=True)
    path = out_dir / "signal_scan.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    winners = [r for r in rows if r["pass"]]
    # Agent 應讀此檔，不要讀 log
    report = {
        "status": "DONE",
        "brand": "tidemark",
        "is_range": [args.is_start, args.is_end],
        "oos_range": [args.oos_start, args.oos_end],
        "n_combos": len(rows),
        "n_winners": len(winners),
        "winners": winners,
        "pass_rule": "IS>0 and OOS>0 and min_oos_trades"
        if args.require_is_positive
        else "OOS>0 and min_oos_trades",
        "detail_path": str(path),
        "note": "IS 虧損組合一律不進 winners（除非 --no-require-is-positive）",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"{notify_prefix()} winners={len(winners)}/{len(rows)} -> {report_path}")
    for w in winners[:10]:
        logger.info(f"  TOP {w['signal']} {w['params']} OOS={w['oos_pnl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
