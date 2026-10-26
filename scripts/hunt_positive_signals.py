#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日勝率獵尋：IS/OOS 雙正且每日勝率>80%（獲利天數/有交易天數）。
順序：wr_hunt（RSI小停利）→ daytrade wide → day 可隔夜 default
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(extra: list[str], workers: int, min_dwr: float) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "scan_signals_fast.py"),
        "--workers",
        str(workers),
        "--force",
        "--min-daily-win-rate",
        str(min_dwr),
        "--min-oos-trades",
        "10",
        "--min-is-trades",
        "10",
        *extra,
    ]
    print(f"[tidemark] hunt launching: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    return json.loads(
        (ROOT / "outputs" / "signal_scan_report.json").read_text(encoding="utf-8")
    )


def main() -> int:
    workers = max(2, (os.cpu_count() or 4) // 2)
    min_dwr = 0.8
    stages = [
        ["--daytrade", "--grid", "wr_hunt"],
        ["--daytrade", "--grid", "wide"],
        [],  # hold overnight default
    ]
    for extra in stages:
        report = _run(extra, workers=workers, min_dwr=min_dwr)
        n = int(report.get("n_winners") or 0)
        print(
            f"[tidemark] stage={extra or ['default']} "
            f"winners={n}/{report.get('n_combos')}",
            flush=True,
        )
        if n > 0:
            print(f"[tidemark] FOUND daily_wr>{min_dwr} — stop", flush=True)
            for w in (report.get("winners") or [])[:12]:
                print(
                    f"  PASS {w.get('signal')} {w.get('params')} "
                    f"stop={w.get('stop_points')} "
                    f"dWR_is={w.get('is_daily_win_rate')} "
                    f"dWR_oos={w.get('oos_daily_win_rate')} "
                    f"IS={w.get('is_pnl')} OOS={w.get('oos_pnl')} "
                    f"days={w.get('is_trade_days')}/{w.get('oos_trade_days')} "
                    f"tr={w.get('is_trades')}/{w.get('oos_trades')}",
                    flush=True,
                )
            return 0
    print(f"[tidemark] no winners with daily_win_rate>{min_dwr}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
