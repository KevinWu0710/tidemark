#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
永豐 SIM 連線／下單探測 + 互補當日分流決策。

預設 dry-run：只測 max_lots 本地紀律，不下真單。
加 --live-order 才對 SIM 下 1 口限價（仍 simulation=True；禁止 LIVE）。

結果 → outputs/paper_sim_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.complementary_fast import (
    ComplementaryConfig,
    today_complementary_choice,
)
from src.core.config import load_settings, notify_prefix
from src.core.types import TradeAction
from src.data.bars import load_1m_range
from src.trading.sinopac import SinopacFuturesTrader


def _recent_choice(settings: dict) -> dict:
    """用近 10 個交易日 1m cache 算最新分流（含 gap 特徵）。"""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
    bars = load_1m_range(start, end)
    cfg = ComplementaryConfig.from_locked(
        capital=float(settings["trading"]["capital"]),
        point_value=float(settings["trading"]["point_value"]),
        fee_per_side=float(settings["trading"].get("fee_per_side", 20)),
        tax_rate=float(settings["trading"].get("tax_rate", 0.00002)),
        slip_ticks=float(settings.get("backtest", {}).get("slip_ticks", 1)),
    )
    return today_complementary_choice(bars, cfg=cfg)


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark Sinopac SIM probe")
    p.add_argument(
        "--live-order",
        action="store_true",
        help="actually place SIM limit order (still simulation API; never LIVE)",
    )
    p.add_argument("--price", type=float, default=0.0, help="limit price; 0 = try quote")
    p.add_argument(
        "--report",
        default="outputs/paper_sim_probe.json",
        help="result JSON path",
    )
    p.add_argument(
        "--action",
        default="BUY",
        choices=["BUY", "SELL", "CLOSE"],
        help="SIM order action when --live-order (default BUY then COVER)",
    )
    args = p.parse_args()

    settings = load_settings()
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "status": "RUNNING",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "brand": settings["brand"]["name"],
        "paper_strategy": settings.get("trading", {}).get("paper_strategy"),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 互補當日決策（不依賴券商）
    try:
        payload["complementary_today"] = _recent_choice(settings)
    except Exception as e:
        payload["complementary_today"] = {"ok": False, "error": str(e)}

    trader = SinopacFuturesTrader(simulation=True, allow_live=False, max_lots=1)
    connected = trader.connect()
    payload["connected"] = connected
    payload["summary_before"] = trader.summary()

    quote_px = None
    if connected:
        try:
            contract = trader._contract()
            # best-effort snapshot
            snap = None
            try:
                snap = trader.api.snapshots([contract])
            except Exception:
                snap = None
            if snap:
                s0 = snap[0]
                quote_px = float(
                    getattr(s0, "close", 0)
                    or getattr(s0, "last_price", 0)
                    or getattr(s0, "deal_price", 0)
                    or 0
                )
            payload["quote"] = {
                "contract": str(getattr(contract, "code", trader.contract_code)),
                "price": quote_px,
            }
        except Exception as e:
            payload["quote"] = {"error": str(e)}

    # Always exercise dry-run max_lots path (no broker)
    dry = SinopacFuturesTrader(simulation=True, max_lots=1)
    ok1 = dry.place_tmf(TradeAction.BUY, 20000.0, reason="probe_open", dry_run=True)
    ok2 = dry.place_tmf(TradeAction.BUY, 20000.0, reason="probe_second", dry_run=True)
    ok3 = dry.place_tmf(TradeAction.CLOSE, 20010.0, reason="probe_close", dry_run=True)
    payload["dry_run"] = {
        "open_ok": ok1,
        "second_open_rejected": not ok2,
        "close_ok": ok3,
        "max_lots_enforced": ok1 and (not ok2) and ok3,
        "summary": dry.summary(),
    }

    # 依互補決策做 dry 劇本說明（不自動真開倉）
    choice = (payload.get("complementary_today") or {}).get("choice")
    payload["sim_playbook"] = {
        "choice": choice,
        "note": (
            "orb→關注開盤突破多單；rsi→等 RSI reclaim 後做多；flat→當日不開。"
            " --live-order 仍是探測單，不會依訊號自動跟單。"
        ),
        "would_allow_new_entry": choice in ("orb", "rsi"),
    }

    order_result = None
    if args.live_order:
        if not connected:
            payload["status"] = "DONE"
            payload["error"] = "not_connected"
            report_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            logger.error(f"{notify_prefix()} probe abort: not connected")
            return 1
        px = float(args.price) if args.price > 0 else float(quote_px or 0)
        if px <= 0:
            payload["status"] = "DONE"
            payload["error"] = "need --price > 0 or valid snapshot quote"
            report_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            return 1
        action = TradeAction(args.action)
        placed = trader.place_tmf(
            action, px, reason=f"sim_probe_{action.value.lower()}", dry_run=False
        )
        order_result = {
            "placed": placed,
            "action": action.value,
            "price": px,
            "summary": trader.summary(),
            "complementary_choice_at_probe": choice,
        }
        if placed and action in (TradeAction.BUY, TradeAction.SELL):
            trader.place_tmf(
                TradeAction.CLOSE, px, reason="sim_probe_cover", dry_run=False
            )
            order_result["summary_after_cover"] = trader.summary()

    trader.close()
    payload["order"] = order_result
    payload["status"] = "DONE"
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} paper sim probe done connected={connected} "
        f"choice={choice} max_lots_ok={payload['dry_run']['max_lots_enforced']} "
        f"-> {report_path}"
    )
    return 0 if payload["dry_run"]["max_lots_enforced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
