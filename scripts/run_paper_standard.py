#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
標準紙上覆盤（常態化）：依 settings trading.paper_strategy 跑 IS / OOS / FULL。

現役預設：complementary（gap≥120→orb else rsi，一口）。
可改回 tmf_v1：settings trading.paper_strategy: tmf_v1

結果 → outputs/paper_standard_report.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.core.config import load_settings, notify_prefix

SPLITS = (
    ("IS", "2024-08-01", "2026-03-31", "standard_is"),
    ("OOS", "2026-04-01", "2026-10-22", "standard_oos"),
    ("FULL", "2024-08-01", "2026-10-22", "standard_full"),
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


def _run_tmf_v1(start: str, end: str, tag: str) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_paper_replay.py"),
        "--start",
        start,
        "--end",
        end,
        "--tag",
        tag,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    latest = ROOT / "outputs" / "paper_session_latest.json"
    blob: dict = {}
    if latest.exists():
        try:
            blob = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            blob = {}
    return {
        "tag": tag,
        "range": [start, end],
        "exit_code": proc.returncode,
        "summary": blob.get("summary"),
        "path": blob.get("path"),
        "stderr_tail": (proc.stderr or "")[-500:],
    }


def _run_complementary(start: str, end: str, tag: str, trading: dict, slip: float) -> dict:
    from src.backtest.complementary_fast import (
        ComplementaryConfig,
        run_complementary_fast,
    )
    from src.backtest.metrics import pair_round_trips, summarize
    from src.data.bars import load_1m_range

    bars = load_1m_range(start, end)
    if bars is None or bars.empty:
        return {
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
    s = summarize(result, capital=cap, label=tag)
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
        "kind": "paper_session",
        "saved_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary": summary,
        "fills": result.trades,
        "round_trips": trips,
        "meta": {
            "tag": tag,
            "bar_range": [start, end],
            "n_bars": len(bars),
            "paper_strategy": "complementary",
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
            {
                "status": "DONE",
                "path": str(path),
                "summary": summary,
                "meta": payload["meta"],
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return {
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
    paper_strategy = str(trading.get("paper_strategy") or "complementary").strip().lower()
    prefix = notify_prefix(settings)

    report_path = ROOT / "outputs" / "paper_standard_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "paper_strategy": paper_strategy,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"{prefix} paper standard IS/OOS/FULL start strategy={paper_strategy}")

    rows = []
    for name, start, end, tag in SPLITS:
        logger.info(f"{prefix} paper standard {name} {start}..{end}")
        if paper_strategy in ("complementary", "comp", "orb_rsi"):
            row = _run_complementary(start, end, tag, trading, slip)
            strat_label = "complementary_gap120_orb_else_rsi"
        elif paper_strategy in ("tmf_v1", "rsi", "rsi28"):
            row = _run_tmf_v1(start, end, tag)
            strat_label = "tmf_v1_locked_rsi28"
        else:
            logger.error(f"{prefix} unknown paper_strategy={paper_strategy}")
            return 1
        row["split"] = name
        rows.append(row)

    ok = all(r.get("exit_code") == 0 for r in rows)
    both_pos = all(
        (r.get("summary") or {}).get("net_pnl", 0) > 0
        for r in rows
        if r.get("split") in ("IS", "OOS")
    )
    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "paper_strategy": paper_strategy,
        "strategy": strat_label,
        "splits": rows,
        "pass_paper_standard": bool(ok and both_pos),
        "note": (
            "常態紙上覆盤：IS/OOS 淨利皆>0 且三輪 exit=0 算過。"
            "主路徑由 trading.paper_strategy 決定（預設 complementary）。"
            "baseline 對照可設 paper_strategy: tmf_v1。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{prefix} paper standard done pass={payload['pass_paper_standard']} "
        f"strategy={strat_label} -> {report_path}"
    )
    return 0 if payload["pass_paper_standard"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
