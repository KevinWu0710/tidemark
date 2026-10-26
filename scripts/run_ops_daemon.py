#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全自動 ops daemon（Docker 常駐）：

  盤中 Paper 模擬 → 晚間覆盤 → 寫 outputs/ops/

用法：
  python scripts/run_ops_daemon.py
  python scripts/run_ops_daemon.py --once-evening --date 2026-10-16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.ops.daily_runner import DailyOpsRunner, load_ops_config


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark ops daemon")
    p.add_argument("--poll", type=float, default=20.0, help="主迴圈秒數")
    p.add_argument(
        "--once-evening",
        action="store_true",
        help="只跑今晚覆盤後退出（測試用）",
    )
    p.add_argument("--date", default="", help="覆盤日期 YYYY-MM-DD")
    args = p.parse_args()

    runner = DailyOpsRunner(load_ops_config())
    if args.once_evening:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        date = args.date or datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d")
        path = runner._run_evening_review(date)
        print(f"[tidemark] evening review -> {path}")
        return 0

    runner.run_forever(poll_seconds=float(args.poll))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
