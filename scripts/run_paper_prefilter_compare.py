#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
紙上對照：現役鎖 vs 日盤事前過濾候選。

過濾（只管日盤進場；夜盤現役不做單）：
  前一夜盤 ≤ -120 點 → 日盤開盤後 60 分不進
  且僅 11:15 前（開盤後 ≤150 分）可進

不改 settings 鎖；結果 → outputs/paper_prefilter_compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.fast_signals import (
    FastSignalConfig,
    _long_only_tp_loop,
    rsi_tp_long_entry_mask,
    run_rsi_tp_fast,
)
from src.backtest.metrics import passes_session_daily_win_rate, summarize
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range
from src.strategy.day_prefilters import DayPrefilterConfig, entry_allow_mask

SPLITS = (
    ("IS", "2024-08-01", "2026-03-31"),
    ("OOS", "2026-04-01", "2026-10-22"),
    ("FULL", "2024-08-01", "2026-10-22"),
)


def _cfg(trading: dict, slip: float, params: dict) -> FastSignalConfig:
    return FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=tuple(params.get("sessions") or ["day"]),
        stop_points=float(params.get("stop_points") or 300),
        max_daily_loss=float(params.get("max_daily_loss") or 0),
        flatten_eod=bool(params.get("flatten_eod") or params.get("flatten_off_session")),
        allow_short=False,
    )


def _score(result, capital: float) -> dict[str, Any]:
    s = summarize(result, capital=capital, label="x")
    day = (s.get("by_session") or {}).get("day") or {}
    night = (s.get("by_session") or {}).get("night") or {}
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "max_drawdown": s["max_drawdown"],
        "day_win_rate": day.get("win_rate"),
        "day_daily_win_rate": day.get("daily_win_rate"),
        "day_n_trade_days": day.get("n_trade_days"),
        "night_win_rate": night.get("win_rate"),
        "night_daily_win_rate": night.get("daily_win_rate"),
        "night_n_trade_days": night.get("n_trade_days"),
        "by_session": s.get("by_session"),
    }


def _run_baseline(bars, cfg: FastSignalConfig, params: dict):
    return run_rsi_tp_fast(
        bars,
        int(params.get("rsi_period", 28)),
        float(params.get("rsi_low", 20)),
        cfg,
        take_profit_points=float(params.get("take_profit_points", 40)),
    )


def _run_filtered(bars, cfg: FastSignalConfig, params: dict, pcfg: DayPrefilterConfig):
    raw = rsi_tp_long_entry_mask(
        bars, int(params.get("rsi_period", 28)), float(params.get("rsi_low", 20))
    )
    allow = entry_allow_mask(bars, pcfg)
    # 僅日盤 bar 套用 allow；非日盤本來就不該進（sessions=day）
    if "session" in bars.columns:
        is_day = bars["session"].astype(str).values == "day"
    else:
        is_day = allow  # fallback
    entry = raw & allow & is_day
    return _long_only_tp_loop(
        bars,
        cfg,
        entry,
        take_profit_points=float(params.get("take_profit_points", 40)),
        reason_entry="rsi_tp_long_prefilter",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="compare lock vs day prefilter on paper path")
    p.add_argument("--min-daily-win-rate", type=float, default=0.8)
    args = p.parse_args()
    min_dwr = float(args.min_daily_win_rate)

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    capital = float(trading["capital"])
    strat = next(s for s in settings["strategies"] if s.get("id") == "tmf_v1")
    params = dict(strat.get("params") or {})

    pcfg = DayPrefilterConfig(
        enabled=True,
        night_ret_le=float(params.get("prefilter_night_ret_le", -120)),
        skip_open_mins=int(params.get("prefilter_skip_open_mins", 60)),
        max_entry_mins=int(params.get("prefilter_max_entry_mins", 150)),
    )

    report_path = ROOT / "outputs" / "paper_prefilter_compare.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} paper prefilter compare "
        f"night<={pcfg.night_ret_le} skip={pcfg.skip_open_mins}m "
        f"max_entry={pcfg.max_entry_mins}m (day only)"
    )

    cfg = _cfg(trading, slip, params)
    splits: dict[str, Any] = {}
    for name, start, end in SPLITS:
        bars = load_1m_range(start, end)
        base = _score(_run_baseline(bars, cfg, params), capital)
        filt = _score(_run_filtered(bars, cfg, params, pcfg), capital)
        splits[name] = {
            "range": [start, end],
            "baseline": base,
            "prefilter": filt,
            "delta_pnl": round(filt["net_pnl"] - base["net_pnl"], 2),
            "delta_day_dwr": round(
                float(filt["day_daily_win_rate"] or 0)
                - float(base["day_daily_win_rate"] or 0),
                4,
            ),
            "delta_dd": round(
                float(base["max_drawdown"] or 0) - float(filt["max_drawdown"] or 0), 2
            ),
            "baseline_dwr_pass": passes_session_daily_win_rate(
                base.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
            ),
            "prefilter_dwr_pass": passes_session_daily_win_rate(
                filt.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
            ),
        }
        logger.info(
            f"{notify_prefix()} {name} base={base['net_pnl']} "
            f"filt={filt['net_pnl']} Δ={splits[name]['delta_pnl']} "
            f"day_dwr {base['day_daily_win_rate']}→{filt['day_daily_win_rate']}"
        )

    is_ok = (
        splits["IS"]["prefilter"]["net_pnl"] > 0
        and splits["OOS"]["prefilter"]["net_pnl"] > 0
        and splits["IS"]["prefilter_dwr_pass"]
        and splits["OOS"]["prefilter_dwr_pass"]
    )
    better = (
        splits["IS"]["delta_pnl"] >= 0
        and splits["OOS"]["delta_pnl"] >= 0
    ) or (
        splits["OOS"]["delta_pnl"] > 0
        and splits["IS"]["delta_pnl"] > -500
        and splits["OOS"]["delta_dd"] >= 0
    )

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "day_session_entries_only",
        "note": (
            "日盤進場過濾；夜盤現役不做單所以 night_* 應為空。"
            "勝率日夜分開；不過關不改 settings 鎖。"
        ),
        "prefilter": {
            "night_ret_le": pcfg.night_ret_le,
            "skip_open_mins": pcfg.skip_open_mins,
            "max_entry_mins": pcfg.max_entry_mins,
            "meaning": "夜盤≤-120→日盤開盤60分不進；日盤僅11:15前可進",
        },
        "min_daily_win_rate": min_dwr,
        "splits": splits,
        "pass_prefilter_candidate": bool(is_ok),
        "recommend_enable": bool(is_ok and better),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} paper prefilter compare done "
        f"pass={payload['pass_prefilter_candidate']} "
        f"recommend={payload['recommend_enable']} -> {report_path}"
    )
    return 0 if is_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
