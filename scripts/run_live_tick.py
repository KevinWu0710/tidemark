#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
盤中即時 tick 觸發（互補 → 一口）。

模式：
  --replay-1m   用本地 1m cache 依序關棒（離線驗證，預設）
  --subscribe   連永豐 SIM 訂閱 TickFOPv1（盤中真路徑）
  --place-sim   對 SIM 真下單（預設只 Paper；禁 LIVE）

用法：
  python scripts/run_live_tick.py --date 2026-10-16
  python scripts/run_live_tick.py --subscribe --max-seconds 120
  python scripts/run_live_tick.py --subscribe --place-sim   # 待可探測後

結果 → outputs/live_tick_YYYYMMDD.json + outputs/live_tick_latest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from loguru import logger

from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range
from src.trading.live_tick_loop import LiveTickConfig, LiveTickLoop

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


def _write(date: str, payload: dict[str, Any]) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"live_tick_{date.replace('-', '')}.json"
    latest = OUT_DIR / "live_tick_latest.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    path.write_text(text, encoding="utf-8")
    latest.write_text(
        json.dumps(
            {
                "status": payload.get("status"),
                "path": str(path),
                "date": date,
                "summary": payload.get("summary"),
                "choice": payload.get("choice"),
                "n_ticks": payload.get("n_ticks"),
                "n_bars_closed": payload.get("n_bars_closed"),
                "n_events": len(payload.get("events") or []),
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return path, latest


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark live tick -> complementary")
    p.add_argument("--date", default="", help="YYYY-MM-DD；預設今天")
    p.add_argument(
        "--subscribe",
        action="store_true",
        help="訂閱永豐 SIM TickFOPv1；未指定則用本地 1m 重播關棒",
    )
    p.add_argument("--place-sim", action="store_true", help="SIM 真下單")
    p.add_argument(
        "--connect",
        action="store_true",
        help="replay 時可選連 SIM dry mirror",
    )
    p.add_argument(
        "--session",
        default="day",
        choices=("day", "night", "all"),
        help="只處理該盤（預設 day）",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0,
        help="subscribe 最長秒數；0=直到日盤結束或 Ctrl+C",
    )
    p.add_argument(
        "--seed-days",
        type=int,
        default=14,
        help="seed 往前取幾日 1m（RSI／跳空）",
    )
    p.add_argument(
        "--risk-seconds",
        type=int,
        default=10,
        help="持倉中每幾秒檢查一次停損停利風控（預設 10 秒）",
    )
    args = p.parse_args()

    settings = load_settings()
    prefix = notify_prefix(settings)
    date = args.date or datetime.now().strftime("%Y-%m-%d")

    if args.place_sim:
        logger.warning(
            f"{prefix} place_sim=ON：將對 SIM 送限價（仍非 LIVE）。"
        )

    connect_sim = bool(args.subscribe or args.place_sim or args.connect)
    cfg = LiveTickConfig(
        date=date,
        dry_run=not args.place_sim,
        place_sim=bool(args.place_sim),
        connect_sim=connect_sim,
        session_filter=str(args.session),
        risk_check_seconds=int(args.risk_seconds),
    )

    start = (
        datetime.strptime(date, "%Y-%m-%d") - timedelta(days=max(1, int(args.seed_days)))
    ).strftime("%Y-%m-%d")
    all_bars = load_1m_range(start, date)
    if all_bars is None or all_bars.empty:
        logger.error(f"{prefix} no 1m seed {start}..{date}")
        _write(date, {"status": "ERROR", "error": "no_bars", "date": date})
        return 1

    import pandas as pd

    all_bars = all_bars.copy()
    all_bars["ts"] = pd.to_datetime(all_bars["ts"])
    # seed：不含當日（當日由 replay／tick 推進，避免偷看整日）
    seed = all_bars[all_bars["ts"].dt.strftime("%Y-%m-%d") < date].copy()
    day = all_bars[all_bars["ts"].dt.strftime("%Y-%m-%d") == date].copy()
    if args.session != "all" and "session" in day.columns:
        day = day[day["session"] == args.session]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "live_tick_latest.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "date": date,
                "mode": "subscribe" if args.subscribe else "replay_1m",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loop = LiveTickLoop(settings, cfg=cfg, seed_bars=seed)

    _n_writes = {"n": 0}

    def _on_event(kind: str, data: dict[str, Any]) -> None:
        # 降頻寫檔：訂閱路徑每次 fill；重播只在 choice／每 50 根
        if kind == "subscribed":
            _write(date, loop.snapshot(status="RUNNING"))
            return
        if kind == "choice":
            _write(date, loop.snapshot(status="RUNNING"))
            return
        if kind == "fill":
            _n_writes["n"] += 1
            if args.subscribe or _n_writes["n"] <= 3:
                _write(date, loop.snapshot(status="RUNNING"))

    loop.on_event = _on_event

    try:
        if args.subscribe:
            logger.info(f"{prefix} live_tick SUBSCRIBE date={date}")
            res = loop.subscribe_and_run(max_seconds=float(args.max_seconds or 0))
        else:
            if day.empty:
                logger.error(f"{prefix} no day 1m for {date} — fetch cache first")
                _write(date, {"status": "ERROR", "error": "no_day_bars", "date": date})
                return 1
            logger.info(
                f"{prefix} live_tick REPLAY_1m date={date} bars={len(day)} seed={len(seed)}"
            )
            res = loop.replay_1m(day)

        path, _ = _write(date, res.payload)
        logger.info(
            f"{prefix} live_tick done date={date} choice={(res.payload.get('choice') or {}).get('choice')} "
            f"events={len(res.payload.get('events') or [])} "
            f"net={((res.payload.get('summary') or {}).get('net_pnl'))} -> {path}"
        )
        return 0 if res.ok else 2
    except KeyboardInterrupt:
        loop.stop()
        path, _ = _write(date, loop.snapshot(status="STOPPED"))
        logger.info(f"{prefix} live_tick stopped -> {path}")
        return 0
    finally:
        loop.close()


if __name__ == "__main__":
    raise SystemExit(main())
