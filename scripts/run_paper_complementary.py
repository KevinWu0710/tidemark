#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真互補分流紙上覆盤：gap≥120→orb else rsi，一口帳本，IS / OOS / FULL。

結果 → outputs/paper_complementary_report.json
各段 session → outputs/paper_sessions/comp_*.json
鎖規則 → outputs/best_complementary_lock.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.complementary_fast import (
    LOCKED_COMPLEMENTARY_RULE,
    ComplementaryConfig,
    run_complementary_fast,
)
from src.backtest.metrics import pair_round_trips, summarize
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range

REPORT_PATH = ROOT / "outputs" / "paper_complementary_report.json"
LOCK_PATH = ROOT / "outputs" / "best_complementary_lock.json"

SPLITS = (
    ("IS", "2024-08-01", "2026-03-31", "comp_is"),
    ("OOS", "2026-04-01", "2026-10-22", "comp_oos"),
    ("FULL", "2024-08-01", "2026-10-22", "comp_full"),
)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _run_split(
    name: str,
    start: str,
    end: str,
    tag: str,
    trading: dict[str, Any],
    slip: float,
) -> dict[str, Any]:
    bars = load_1m_range(start, end)
    if bars is None or bars.empty:
        return {
            "split": name,
            "tag": tag,
            "range": [start, end],
            "exit_code": 1,
            "error": "no bars",
        }

    cfg = ComplementaryConfig.from_locked(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=slip,
    )
    result, meta = run_complementary_fast(bars, cfg)
    cap = float(trading["capital"])
    s = summarize(result, capital=cap, label=name)
    trips = pair_round_trips(result.trades)

    summary = {
        "session_id": tag,
        "instrument": trading.get("instrument", "TMF"),
        "capital": cap,
        "cash": round(float(result.final_equity), 2),
        "equity": round(float(result.final_equity), 2),
        "net_pnl": round(float(s["net_pnl"]), 2),
        "n_fills": len(result.trades),
        "n_round_trips": int(s["n_round_trips"]),
        "n_wins": int(s["n_wins"]),
        "n_losses": int(s["n_losses"]),
        "win_rate": s.get("win_rate"),
        "day_win_rate": s.get("day_win_rate"),
        "day_daily_win_rate": s.get("day_daily_win_rate"),
        "daily_win_rate": s.get("daily_win_rate"),
        "max_drawdown": round(float(s.get("max_drawdown") or 0), 2),
        "engine": "complementary_orb_rsi",
        "rule_id": cfg.rule.id,
        "n_choice_orb": meta.get("n_choice_orb"),
        "n_choice_rsi": meta.get("n_choice_rsi"),
        "n_choice_flat": meta.get("n_choice_flat"),
    }

    out_dir = ROOT / "outputs" / "paper_sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    payload = {
        "status": "DONE",
        "brand": "tidemark",
        "kind": "paper_session_complementary",
        "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary": summary,
        "fills": result.trades,
        "round_trips": trips,
        "meta": {
            "tag": tag,
            "split": name,
            "bar_range": [start, end],
            "n_bars": len(bars),
            "complementary": meta,
            "max_lots": 1,
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    latest = ROOT / "outputs" / "paper_session_latest.json"
    latest.write_text(
        json.dumps(
            {"status": "DONE", "path": str(path), "summary": summary, "meta": payload["meta"]},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    return {
        "split": name,
        "tag": tag,
        "range": [start, end],
        "exit_code": 0,
        "summary": summary,
        "path": str(path),
    }


def main() -> int:
    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    prefix = notify_prefix(settings)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {"status": "RUNNING", "updated_at": datetime.now().isoformat(timespec="seconds")},
            indent=2,
        ),
        encoding="utf-8",
    )
    LOCK_PATH.write_text(
        json.dumps(
            {
                "status": "LOCKED",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "rule": LOCKED_COMPLEMENTARY_RULE.to_dict(),
                "tracks": {
                    "orb": "tmf_daily_trend open_gap_orb TP200/SL50",
                    "rsi": "tmf_v1 RSI28 reclaim20 TP40/stop300 after 09:00",
                },
                "max_lots": 1,
                "source": "research_complementary_selector locked_by_net_pnl",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(f"{prefix} paper complementary IS/OOS/FULL start")
    rows = []
    for name, start, end, tag in SPLITS:
        logger.info(f"{prefix} complementary {name} {start}..{end}")
        rows.append(_run_split(name, start, end, tag, trading, slip))

    ok = all(r.get("exit_code") == 0 for r in rows)
    both_pos = all(
        (r.get("summary") or {}).get("net_pnl", 0) > 0
        for r in rows
        if r.get("split") in ("IS", "OOS")
    )

    compare_tmf_v1: dict[str, Any] | None = None
    std_path = ROOT / "outputs" / "paper_standard_report.json"
    if std_path.exists():
        try:
            std = json.loads(std_path.read_text(encoding="utf-8"))
            by_split = {
                r.get("split"): (r.get("summary") or {}).get("net_pnl")
                for r in (std.get("splits") or [])
            }
            compare_tmf_v1 = {
                "source": str(std_path),
                "strategy": std.get("strategy"),
                "net_pnl": by_split,
                "delta_vs_tmf_v1": {
                    r["split"]: round(
                        float((r.get("summary") or {}).get("net_pnl") or 0)
                        - float(by_split.get(r["split"]) or 0),
                        2,
                    )
                    for r in rows
                    if r.get("exit_code") == 0
                },
            }
        except Exception:
            compare_tmf_v1 = {"error": "failed to parse paper_standard_report"}

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy": "complementary_gap120_orb_else_rsi",
        "rule": LOCKED_COMPLEMENTARY_RULE.to_dict(),
        "lock_path": str(LOCK_PATH),
        "max_lots": 1,
        "splits": rows,
        "pass_paper_complementary": bool(ok and both_pos),
        "compare_vs_paper_standard": compare_tmf_v1,
        "note": (
            "IS/OOS 淨利皆>0 且三輪 exit=0 算過。"
            "與 paper_standard（純 tmf_v1）並列比較，不自動取代主鎖。"
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    logger.info(
        f"{prefix} paper complementary done pass={payload['pass_paper_complementary']} "
        f"-> {REPORT_PATH}"
    )
    return 0 if payload["pass_paper_complementary"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
