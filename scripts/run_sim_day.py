#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日流程自動化（互補分流 → 一口）。

預設 dry-run：Paper 記帳 + Sinopac dry（不送券商真單）。
之後可加 --place-sim 對永豐 SIM 真下限價（仍 simulation；禁 LIVE）。

用法：
  python scripts/run_sim_day.py --date 2026-10-22
  python scripts/run_sim_day.py --watch --interval 60
  python scripts/run_sim_day.py --date 2026-10-22 --connect          # dry + 連 SIM
  python scripts/run_sim_day.py --date 2026-10-22 --place-sim        # 需 SIM 可下單時

結果 → outputs/sim_day_YYYYMMDD.json + outputs/sim_day_latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range
from src.data.sessions import label_session
from src.trading.sim_day_loop import SimDayConfig, SimDayLoop

OUT_DIR = ROOT / "outputs"


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _write_reports(date: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"sim_day_{date.replace('-', '')}.json"
    latest = OUT_DIR / "sim_day_latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(text, encoding="utf-8")
    latest.write_text(
        json.dumps(
            {"status": payload.get("status"), "path": str(path), "summary": payload.get("summary"), "choice": payload.get("choice"), "date": date},
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return path, latest


def _load_bars_for_day(date: str):
    # 多取前段供 RSI／跳空特徵
    start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=14)).strftime("%Y-%m-%d")
    return load_1m_range(start, date)


def _session_open_now() -> bool:
    """粗判：本地時間是否落在日盤（08:45–13:45）。"""
    now = datetime.now()
    sess = label_session(pd.Timestamp(now))
    if sess != "day":
        return False
    mins = now.hour * 60 + now.minute
    return (8 * 60 + 45) <= mins <= (13 * 60 + 45)


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark SIM day automation (complementary)")
    p.add_argument("--date", default="", help="YYYY-MM-DD；預設今天")
    p.add_argument(
        "--watch",
        action="store_true",
        help="盤中輪詢；每輪重讀 cache 並只執行新意圖",
    )
    p.add_argument("--interval", type=int, default=60, help="watch 秒數（預設 60）")
    p.add_argument(
        "--place-sim",
        action="store_true",
        help="對永豐 SIM API 真下單（預設只 Paper dry-run）",
    )
    p.add_argument(
        "--connect",
        action="store_true",
        help="連 Sinopac SIM（dry mirror）；預設不連，純 Paper",
    )
    p.add_argument(
        "--no-connect",
        action="store_true",
        help=argparse.SUPPRESS,  # 相容舊旗標；預設已是不連
    )
    p.add_argument("--max-loops", type=int, default=0, help="watch 最多幾輪；0=直到日盤結束")
    args = p.parse_args()

    settings = load_settings()
    prefix = notify_prefix(settings)
    date = args.date or datetime.now().strftime("%Y-%m-%d")

    if args.place_sim:
        logger.warning(
            f"{prefix} place_sim=ON：將對 SIM API 送限價（仍非 LIVE）。確認測試環境後再繼續。"
        )

    connect_sim = bool(args.place_sim or args.connect) and not args.no_connect
    cfg = SimDayConfig(
        date=date,
        dry_run=not args.place_sim,
        place_sim=bool(args.place_sim),
        connect_sim=connect_sim,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    running = {
        "status": "RUNNING",
        "date": date,
        "watch": bool(args.watch),
        "place_sim": bool(args.place_sim),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "sim_day_latest.json").write_text(
        json.dumps(running, indent=2), encoding="utf-8"
    )

    loop = SimDayLoop(settings, cfg=cfg)

    def _one() -> int:
        bars = _load_bars_for_day(date)
        if bars is None or bars.empty:
            logger.error(f"{prefix} no 1m bars for {date} — fetch cache first")
            payload = {
                "status": "ERROR",
                "date": date,
                "error": "no_bars",
                "place_sim": cfg.place_sim,
            }
            _write_reports(date, payload)
            return 1
        res = loop.run_once(bars)
        path, _ = _write_reports(date, res.payload)
        logger.info(
            f"{prefix} sim_day done date={date} choice={res.payload.get('choice', {}).get('choice')} "
            f"net={((res.payload.get('summary') or {}).get('net_pnl'))} dry={res.payload.get('dry_run')} -> {path}"
        )
        return 0 if res.ok else 2

    try:
        if not args.watch:
            return _one()

        # watch：同一 loop 實例保留已執行意圖，避免重下
        n = 0
        while True:
            n += 1
            rc = _one()
            if args.max_loops and n >= args.max_loops:
                logger.info(f"{prefix} watch stop max_loops={args.max_loops}")
                return rc
            if not _session_open_now() and date == datetime.now().strftime("%Y-%m-%d"):
                logger.info(f"{prefix} watch stop: day session ended")
                return rc
            # 歷史 --date 的 watch 無意義，跑一輪即停
            if date != datetime.now().strftime("%Y-%m-%d"):
                logger.info(f"{prefix} watch with historical --date runs once")
                return rc
            time.sleep(max(5, int(args.interval)))
    finally:
        loop.close()


if __name__ == "__main__":
    raise SystemExit(main())
