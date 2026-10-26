#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日頻／覆蓋率研究：日+夜、可多可空；對照現役高日勝率鎖。

上一輪（min_cov=0.8、不卡勝率）0 過關。本輪：
  - 日勝率門檻降到 >70%（低於現役鎖的 80%）
  - 覆蓋率改「階梯」回報（不硬鎖 0.8）
  - 加訊號族：rsi_tp／macd_tp／torb（保留原高頻族）

過關（某覆蓋率階梯）：
  IS>0 且 OOS>0
  且 IS/OOS daily_win_rate > --min-daily-win-rate
  且 IS/OOS day_coverage >= 該階梯
  且筆數門檻

不過關不改 tmf_v1。結果 → outputs/daily_frequency_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.fast_breakout import FastBreakoutConfig, run_breakout_fast
from src.backtest.fast_signals import (
    FastSignalConfig,
    run_ema_fast,
    run_macd_tp_fast,
    run_rsi_fast,
    run_rsi_tp_fast,
    run_sma_cross_fast,
    run_torb_fast,
)
from src.backtest.metrics import (
    daily_pnl_stats,
    pair_round_trips,
    passes_session_daily_win_rate,
    summarize,
)
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range


SESSIONS_DN = ("day", "night")
DEFAULT_TIERS = (0.15, 0.3, 0.5, 0.8)


def _available_dates(bars: pd.DataFrame) -> set[str]:
    if bars is None or bars.empty:
        return set()
    df = bars
    if "session" in df.columns:
        df = df[df["session"].astype(str).isin(["day", "night"])]
    ts = pd.to_datetime(df["ts"])
    return set(ts.dt.strftime("%Y-%m-%d").tolist())


def _cfg(
    trading: dict,
    slip: float,
    *,
    stop: float,
    sessions: tuple[str, ...],
    allow_short: bool = True,
    flatten: bool = True,
) -> FastSignalConfig:
    return FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=sessions,
        stop_points=float(stop),
        max_daily_loss=0.0,
        flatten_eod=flatten,
        allow_short=allow_short,
    )


def _score(result, capital: float, available: set[str]) -> dict[str, Any]:
    s = summarize(result, capital=capital, label="x")
    d = daily_pnl_stats(pair_round_trips(result.trades))
    n_avail = len(available)
    n_td = int(d["n_trade_days"] or 0)
    cov = round(n_td / n_avail, 4) if n_avail else 0.0
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "win_rate": s["win_rate"],
        "n_trade_days": n_td,
        "n_available_days": n_avail,
        "day_coverage": cov,
        "daily_win_rate": s.get("day_daily_win_rate"),  # 日盤；勿用混算
        "night_daily_win_rate": s.get("night_daily_win_rate"),
        "by_session": s.get("by_session"),
        "max_drawdown": s["max_drawdown"],
    }


def _sessions_of(params: dict) -> tuple[str, ...]:
    raw = params.get("sessions") or list(SESSIONS_DN)
    return tuple(str(x) for x in raw)


def _combo_grid() -> list[dict[str, Any]]:
    """高頻族 + WR 取向族（體積可控，約 300 組）。"""
    combos: list[dict[str, Any]] = []

    # --- A: 原高頻（多空日夜）---
    stops_hf = (0.0, 150.0, 300.0)
    for stop in stops_hf:
        for p in (5, 8, 10, 14):
            for lo, hi in ((25.0, 75.0), (30.0, 70.0)):
                combos.append(
                    {
                        "signal": "rsi_mr",
                        "params": {
                            "rsi_period": p,
                            "rsi_low": lo,
                            "rsi_high": hi,
                            "sessions": list(SESSIONS_DN),
                            "allow_short": True,
                        },
                        "stop_points": stop,
                    }
                )
        for fast, slow in ((5, 15), (5, 20), (8, 30), (10, 40)):
            combos.append(
                {
                    "signal": "sma_cross",
                    "params": {
                        "sma_fast": fast,
                        "sma_slow": slow,
                        "sessions": list(SESSIONS_DN),
                        "allow_short": True,
                    },
                    "stop_points": stop,
                }
            )
        for ema in (8, 12, 20):
            combos.append(
                {
                    "signal": "ema_mom",
                    "params": {
                        "ema_period": ema,
                        "sessions": list(SESSIONS_DN),
                        "allow_short": True,
                    },
                    "stop_points": stop,
                }
            )
        for lb in (10, 20, 30):
            combos.append(
                {
                    "signal": "breakout",
                    "params": {"lookback": lb, "sessions": list(SESSIONS_DN)},
                    "stop_points": stop,
                }
            )

    # --- B: RSI 小停利（現役鎖同族，門檻放寬＋日夜）---
    for sessions in (["day"], list(SESSIONS_DN)):
        for period in (21, 28, 35):
            for low in (15.0, 20.0, 25.0):
                for tp in (30.0, 40.0, 50.0, 60.0):
                    for stop in (200.0, 300.0):
                        combos.append(
                            {
                                "signal": "rsi_tp",
                                "params": {
                                    "rsi_period": period,
                                    "rsi_low": low,
                                    "take_profit": tp,
                                    "sessions": sessions,
                                    "allow_short": False,
                                },
                                "stop_points": stop,
                            }
                        )

    # --- C: MACD + 小停利 ---
    for sessions in (["day"], list(SESSIONS_DN)):
        for mode in ("hist_cross_up", "line_cross_up"):
            for tp in (30.0, 40.0, 50.0):
                for stop in (200.0, 300.0):
                    combos.append(
                        {
                            "signal": "macd_tp",
                            "params": {
                                "mode": mode,
                                "take_profit": tp,
                                "sessions": sessions,
                                "allow_short": False,
                            },
                            "stop_points": stop,
                        }
                    )

    # --- D: TORB 多空日夜 ---
    for probe in (5, 15, 30, 45):
        for tp in (0.0, 40.0, 60.0):
            for stop in (150.0, 300.0):
                combos.append(
                    {
                        "signal": "torb",
                        "params": {
                            "probe_minutes": probe,
                            "take_profit": tp,
                            "sessions": list(SESSIONS_DN),
                            "allow_short": True,
                            "once_per_day": True,
                        },
                        "stop_points": stop,
                    }
                )

    return combos


def _run_one(bars, spec: dict[str, Any], trading: dict, slip: float):
    stop = float(spec["stop_points"])
    name = spec["signal"]
    params = spec["params"]
    sessions = _sessions_of(params)
    allow_short = bool(params.get("allow_short", True))

    if name == "breakout":
        cfg = FastBreakoutConfig(
            lookback=int(params["lookback"]),
            capital=float(trading["capital"]),
            point_value=float(trading["point_value"]),
            fee_per_side=float(trading.get("fee_per_side", 20)),
            tax_rate=float(trading.get("tax_rate", 0.00002)),
            slip_ticks=float(slip),
            sessions=sessions,
            stop_points=stop,
            max_daily_loss=0.0,
            flatten_eod=True,
        )
        return run_breakout_fast(bars, cfg)

    cfg = _cfg(
        trading,
        slip,
        stop=stop,
        sessions=sessions,
        allow_short=allow_short,
        flatten=True,
    )
    if name == "rsi_mr":
        return run_rsi_fast(
            bars,
            int(params["rsi_period"]),
            float(params["rsi_low"]),
            float(params["rsi_high"]),
            cfg,
        )
    if name == "sma_cross":
        return run_sma_cross_fast(
            bars, int(params["sma_fast"]), int(params["sma_slow"]), cfg
        )
    if name == "ema_mom":
        return run_ema_fast(bars, int(params["ema_period"]), cfg)
    if name == "rsi_tp":
        return run_rsi_tp_fast(
            bars,
            int(params["rsi_period"]),
            float(params["rsi_low"]),
            cfg,
            take_profit_points=float(params["take_profit"]),
        )
    if name == "macd_tp":
        return run_macd_tp_fast(
            bars,
            cfg,
            take_profit_points=float(params["take_profit"]),
            mode=str(params.get("mode", "hist_cross_up")),
        )
    if name == "torb":
        return run_torb_fast(
            bars,
            int(params["probe_minutes"]),
            cfg,
            take_profit_points=float(params.get("take_profit", 0.0)),
            once_per_day=bool(params.get("once_per_day", True)),
        )
    raise ValueError(name)


def _edge_ok(is_s: dict, oos_s: dict, min_dwr: float) -> bool:
    return bool(
        is_s["net_pnl"] > 0
        and oos_s["net_pnl"] > 0
        and passes_session_daily_win_rate(
            is_s.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
        )
        and passes_session_daily_win_rate(
            oos_s.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
        )
        and is_s["n_round_trips"] >= 30
        and oos_s["n_round_trips"] >= 15
    )


def _pass_at_cov(is_s: dict, oos_s: dict, min_dwr: float, min_cov: float) -> bool:
    return bool(
        _edge_ok(is_s, oos_s, min_dwr)
        and is_s["day_coverage"] >= min_cov
        and oos_s["day_coverage"] >= min_cov
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="tidemark daily-freq: WR>70% + coverage tiers + new families"
    )
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument(
        "--min-daily-win-rate",
        type=float,
        default=0.7,
        help="IS/OOS 日勝率下限（嚴格大於；預設 0.7，低於現役鎖 0.8）",
    )
    p.add_argument(
        "--coverage-tiers",
        default="0.15,0.3,0.5,0.8",
        help="逗號分隔覆蓋率階梯，各階獨立計過關數",
    )
    args = p.parse_args()
    min_dwr = float(args.min_daily_win_rate)
    tiers = tuple(
        float(x.strip()) for x in str(args.coverage_tiers).split(",") if x.strip()
    ) or DEFAULT_TIERS

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    capital = float(trading["capital"])
    is_bars = load_1m_range(args.is_start, args.is_end)
    oos_bars = load_1m_range(args.oos_start, args.oos_end)
    is_avail = _available_dates(is_bars)
    oos_avail = _available_dates(oos_bars)

    combos = _combo_grid()
    report_path = ROOT / "outputs" / "daily_frequency_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "n_combos": len(combos),
                "min_daily_win_rate": min_dwr,
                "coverage_tiers": list(tiers),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} daily-freq v2 n={len(combos)} "
        f"min_dwr={min_dwr} tiers={list(tiers)}"
    )

    lock_cfg = FastSignalConfig(
        capital=capital,
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=slip,
        sessions=("day",),
        stop_points=300.0,
        flatten_eod=True,
        allow_short=False,
    )
    baseline = {
        "id": "locked_rsi28_day_only",
        "note": "現役鎖：日盤只做多；日勝率門檻曾用 0.8",
        "IS": _score(
            run_rsi_tp_fast(is_bars, 28, 20.0, lock_cfg, take_profit_points=40.0),
            capital,
            is_avail,
        ),
        "OOS": _score(
            run_rsi_tp_fast(oos_bars, 28, 20.0, lock_cfg, take_profit_points=40.0),
            capital,
            oos_avail,
        ),
    }

    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(combos):
        is_s = _score(_run_one(is_bars, spec, trading, slip), capital, is_avail)
        oos_s = _score(_run_one(oos_bars, spec, trading, slip), capital, oos_avail)
        edge = _edge_ok(is_s, oos_s, min_dwr)
        tier_pass = {str(t): _pass_at_cov(is_s, oos_s, min_dwr, t) for t in tiers}
        rows.append(
            {
                "signal": spec["signal"],
                "params": spec["params"],
                "stop_points": spec["stop_points"],
                "IS": is_s,
                "OOS": oos_s,
                "edge_ok": edge,
                "tier_pass": tier_pass,
                "pass": any(tier_pass.values()),
                "min_coverage": round(
                    min(is_s["day_coverage"], oos_s["day_coverage"]), 4
                ),
            }
        )
        if (i + 1) % 40 == 0 or i + 1 == len(combos):
            logger.info(f"{notify_prefix()} daily-freq {i+1}/{len(combos)}")

    # 有 edge 的：覆蓋率高者優先；否則仍按覆蓋／PnL 排
    rows.sort(
        key=lambda r: (
            r["edge_ok"],
            r["min_coverage"],
            r["OOS"]["net_pnl"],
            r["IS"]["net_pnl"],
        ),
        reverse=True,
    )

    by_tier: dict[str, Any] = {}
    for t in tiers:
        key = str(t)
        wins = [r for r in rows if r["tier_pass"].get(key)]
        by_tier[key] = {
            "n_pass": len(wins),
            "best": wins[0] if wins else None,
            "winners_top5": wins[:5],
        }

    edge_rows = [r for r in rows if r["edge_ok"]]
    best_edge = edge_rows[0] if edge_rows else None
    any_pass = any(v["n_pass"] > 0 for v in by_tier.values())

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "goal": "wr_gt_70_then_max_coverage_day_night_families",
        "sessions_default": list(SESSIONS_DN),
        "min_daily_win_rate": min_dwr,
        "coverage_tiers": list(tiers),
        "pass_rule": (
            f"IS>0 OOS>0 daily_win_rate>{min_dwr} "
            "IS_trades>=30 OOS_trades>=15；各 tier 另要求 day_coverage>=tier"
        ),
        "n_combos": len(combos),
        "n_edge_ok": len(edge_rows),
        "n_pass_any_tier": sum(1 for r in rows if r["pass"]),
        "available_days": {"IS": len(is_avail), "OOS": len(oos_avail)},
        "baseline_rsi_lock": baseline,
        "by_tier": by_tier,
        "best_edge": best_edge,
        "top15_edge_or_cov": rows[:15],
        "note": (
            "日勝率先降到 >70%，再比覆蓋率／換訊號族。"
            "任一 tier 過關才考慮改鎖；否則維持 tmf_v1。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} daily-freq done edge={len(edge_rows)}/{len(combos)} "
        f"tiers={{ {', '.join(f'{k}:{v['n_pass']}' for k, v in by_tier.items())} }} "
        f"-> {report_path}"
    )
    if best_edge:
        b = best_edge
        logger.info(
            f"  BEST_EDGE {b['signal']} {b['params']} stop={b['stop_points']} "
            f"cov={b['IS']['day_coverage']}/{b['OOS']['day_coverage']} "
            f"dwr={b['IS']['daily_win_rate']}/{b['OOS']['daily_win_rate']} "
            f"IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']}"
        )
    return 0 if any_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
