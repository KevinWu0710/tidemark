#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MACD 當主訊號網格掃描（日盤 flatten + 小 TP／寬 stop），
對照現役 RSI28 鎖策略。結果 → outputs/macd_primary_report.json
"""
from __future__ import annotations

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
    run_macd_tp_fast,
    run_rsi_tp_fast,
)
from src.backtest.metrics import daily_pnl_stats, pair_round_trips, summarize
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range


def _cfg(
    trading: dict,
    slip: float,
    *,
    stop: float,
    daily: float,
    flatten: bool = True,
) -> FastSignalConfig:
    return FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=("day",),
        stop_points=float(stop),
        max_daily_loss=float(daily),
        flatten_eod=flatten,
        allow_short=False,
    )


def _score(result, capital: float) -> dict[str, Any]:
    s = summarize(result, capital=capital, label="x")
    d = daily_pnl_stats(pair_round_trips(result.trades))
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "win_rate": s["win_rate"],
        "n_trade_days": d["n_trade_days"],
        "n_win_days": d["n_win_days"],
        "n_loss_days": d["n_loss_days"],
        "daily_win_rate": d["daily_win_rate"],
        "max_drawdown": s["max_drawdown"],
    }


def _pass(is_s: dict, oos_s: dict, min_dwr: float = 0.8) -> bool:
    return bool(
        is_s["net_pnl"] > 0
        and oos_s["net_pnl"] > 0
        and (is_s.get("daily_win_rate") or 0) > min_dwr
        and (oos_s.get("daily_win_rate") or 0) > min_dwr
        and (is_s.get("n_trade_days") or 0) >= 30
        and (oos_s.get("n_trade_days") or 0) >= 15
        and (is_s.get("n_round_trips") or 0) >= 10
        and (oos_s.get("n_round_trips") or 0) >= 10
    )


def main() -> int:
    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    capital = float(trading["capital"])
    is_bars = load_1m_range("2024-08-01", "2026-03-31")
    oos_bars = load_1m_range("2026-04-01", "2026-10-22")

    report_path = ROOT / "outputs" / "macd_primary_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # baseline: locked RSI
    lock_cfg = _cfg(trading, slip, stop=300, daily=0)
    rsi_is = run_rsi_tp_fast(is_bars, 28, 20.0, lock_cfg, take_profit_points=40.0)
    rsi_oos = run_rsi_tp_fast(oos_bars, 28, 20.0, lock_cfg, take_profit_points=40.0)
    baseline = {
        "id": "locked_rsi28_tp40_stop300",
        "signal": "rsi_tp",
        "params": {"rsi_period": 28, "rsi_low": 20, "take_profit_points": 40, "stop_points": 300},
        "IS": _score(rsi_is, capital),
        "OOS": _score(rsi_oos, capital),
        "pass_daily_wr_0.8": _pass(_score(rsi_is, capital), _score(rsi_oos, capital)),
    }

    modes = ("hist_cross_up", "line_cross_up", "hist_up")
    macd_sets = (
        (8, 17, 9),
        (12, 26, 9),
        (5, 35, 5),
        (12, 26, 5),
        (16, 32, 9),
    )
    tps = (20, 30, 40, 50, 60)
    stops = (150, 200, 250, 300)
    dailies = (0.0,)

    combos: list[dict[str, Any]] = []
    for mode in modes:
        for fast, slow, sig in macd_sets:
            if fast >= slow:
                continue
            for tp in tps:
                for stop in stops:
                    if stop < tp * 2:
                        continue
                    for daily in dailies:
                        combos.append(
                            {
                                "mode": mode,
                                "fast": fast,
                                "slow": slow,
                                "signal": sig,
                                "tp": tp,
                                "stop": stop,
                                "daily": daily,
                            }
                        )

    report_path.write_text(
        json.dumps(
            {"status": "RUNNING", "n_combos": len(combos), "updated_at": datetime.now().isoformat()},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"{notify_prefix()} MACD primary scan n={len(combos)}")

    rows: list[dict[str, Any]] = []
    for i, c in enumerate(combos):
        cfg = _cfg(trading, slip, stop=c["stop"], daily=c["daily"])
        is_r = run_macd_tp_fast(
            is_bars,
            cfg,
            take_profit_points=c["tp"],
            fast=c["fast"],
            slow=c["slow"],
            signal=c["signal"],
            mode=c["mode"],
        )
        oos_r = run_macd_tp_fast(
            oos_bars,
            cfg,
            take_profit_points=c["tp"],
            fast=c["fast"],
            slow=c["slow"],
            signal=c["signal"],
            mode=c["mode"],
        )
        is_s, oos_s = _score(is_r, capital), _score(oos_r, capital)
        row = {
            "signal": "macd_tp",
            "params": c,
            "IS": is_s,
            "OOS": oos_s,
            "pass_daily_wr_0.8": _pass(is_s, oos_s),
            "both_pos": is_s["net_pnl"] > 0 and oos_s["net_pnl"] > 0,
        }
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(combos):
            logger.info(f"{notify_prefix()} MACD {i+1}/{len(combos)}")

    rows.sort(
        key=lambda r: (
            r["pass_daily_wr_0.8"],
            r["both_pos"],
            r["OOS"].get("daily_win_rate") or 0,
            r["IS"].get("daily_win_rate") or 0,
            r["OOS"]["net_pnl"],
            r["IS"]["net_pnl"],
        ),
        reverse=True,
    )
    winners = [r for r in rows if r["pass_daily_wr_0.8"]]
    best = rows[0] if rows else None

    def _cmp(label: str, a: dict, b: dict) -> dict:
        return {
            "metric": label,
            "rsi_lock": a,
            "macd_best": b,
            "macd_minus_rsi_pnl_is": round(b["IS"]["net_pnl"] - a["IS"]["net_pnl"], 2),
            "macd_minus_rsi_pnl_oos": round(b["OOS"]["net_pnl"] - a["OOS"]["net_pnl"], 2),
            "macd_minus_rsi_dwr_is": round(
                (b["IS"].get("daily_win_rate") or 0) - (a["IS"].get("daily_win_rate") or 0), 4
            ),
            "macd_minus_rsi_dwr_oos": round(
                (b["OOS"].get("daily_win_rate") or 0) - (a["OOS"].get("daily_win_rate") or 0), 4
            ),
        }

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "pass_rule": "IS>0 OOS>0 daily_wr>0.8 trade_days IS>=30 OOS>=15 trades>=10",
        "n_combos": len(combos),
        "n_pass": len(winners),
        "baseline_rsi_lock": baseline,
        "best_macd": best,
        "winners": winners[:15],
        "top10": rows[:10],
        "comparison": _cmp("best_vs_rsi_lock", baseline, best) if best else None,
        "note": (
            "MACD 主訊號（非 RSI 過濾）；日盤 flatten；與 locked RSI28/TP40/stop300 同區間對照。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        f"{notify_prefix()} MACD primary done pass={len(winners)}/{len(combos)} -> {report_path}"
    )
    if best:
        logger.info(
            f"  BEST {best['params']} dWR_is={best['IS'].get('daily_win_rate')} "
            f"dWR_oos={best['OOS'].get('daily_win_rate')} "
            f"IS={best['IS']['net_pnl']} OOS={best['OOS']['net_pnl']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
