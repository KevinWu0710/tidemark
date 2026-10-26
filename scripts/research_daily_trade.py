#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日頻專用軌道（與 tmf_v1 高勝率鎖分開）。

目標：
  - 日+夜都可交易
  - 覆蓋率盡量高（每個有行情的交易日都盡量有成交）
  - 同一天可多回合（仍 max_lots=1：平倉後可再進）
  - 多／空用不同訊號族，再合成

過關（較鎖放寬）：
  IS>0 OOS>0
  日／夜盤交易日勝率分開 > --min-daily-win-rate（預設 0.55；有成交的 session 才計）
  覆蓋率達各 tier

不過關不 enable tmf_daily_v1。
結果 → outputs/daily_trade_report.json
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
    run_dual_rsi_tp_fast,
    run_ema_fast,
    run_rsi_fast,
    run_rsi_tp_fast,
    run_rsi_tp_short_fast,
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


SESSIONS = ("day", "night")
DEFAULT_TIERS = (0.5, 0.7, 0.9)


def _available_dates(bars: pd.DataFrame) -> set[str]:
    if bars is None or bars.empty:
        return set()
    df = bars
    if "session" in df.columns:
        df = df[df["session"].astype(str).isin(["day", "night"])]
    return set(pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d").tolist())


def _cfg(
    trading: dict,
    slip: float,
    *,
    stop: float,
    sessions: tuple[str, ...] = SESSIONS,
    allow_short: bool = True,
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
        flatten_eod=True,
        allow_short=allow_short,
    )


def _score(result, capital: float, available: set[str]) -> dict[str, Any]:
    s = summarize(result, capital=capital, label="x")
    trips = pair_round_trips(result.trades)
    d = daily_pnl_stats(trips, session=None)  # 覆蓋用：任一盤有成交的曆日
    n_avail = len(available)
    n_td = int(d["n_trade_days"] or 0)
    cov = round(n_td / n_avail, 4) if n_avail else 0.0
    avg_trips = round(len(trips) / n_td, 3) if n_td else 0.0
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "max_drawdown": s["max_drawdown"],
        "n_trade_days": n_td,
        "n_available_days": n_avail,
        "day_coverage": cov,
        "avg_trips_per_trade_day": avg_trips,
        "day_daily_win_rate": s.get("day_daily_win_rate"),
        "night_daily_win_rate": s.get("night_daily_win_rate"),
        "day_win_rate": s.get("day_win_rate"),
        "night_win_rate": s.get("night_win_rate"),
        "by_session": s.get("by_session"),
    }


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
        and is_s["n_round_trips"] >= 40
        and oos_s["n_round_trips"] >= 20
    )


def _pass_tiers(
    is_s: dict, oos_s: dict, min_dwr: float, tiers: tuple[float, ...]
) -> dict[str, bool]:
    edge = _edge_ok(is_s, oos_s, min_dwr)
    return {
        str(t): bool(
            edge
            and is_s["day_coverage"] >= t
            and oos_s["day_coverage"] >= t
        )
        for t in tiers
    }


def _solo_grid() -> list[dict[str, Any]]:
    """密訊號網格：偏日頻、可多空。"""
    combos: list[dict[str, Any]] = []
    stops = (100.0, 150.0, 200.0, 300.0)

    # RSI 雙向 mean-revert（密）
    for stop in stops:
        for p in (4, 5, 6, 8, 10):
            for lo, hi in ((20.0, 80.0), (25.0, 75.0), (30.0, 70.0)):
                combos.append(
                    {
                        "kind": "solo",
                        "signal": "rsi_mr",
                        "params": {
                            "rsi_period": p,
                            "rsi_low": lo,
                            "rsi_high": hi,
                            "sessions": list(SESSIONS),
                            "allow_short": True,
                        },
                        "stop_points": stop,
                    }
                )
        for fast, slow in ((3, 10), (5, 15), (5, 20), (8, 21), (10, 30)):
            combos.append(
                {
                    "kind": "solo",
                    "signal": "sma_cross",
                    "params": {
                        "sma_fast": fast,
                        "sma_slow": slow,
                        "sessions": list(SESSIONS),
                        "allow_short": True,
                    },
                    "stop_points": stop,
                }
            )
        for ema in (5, 8, 12, 20):
            combos.append(
                {
                    "kind": "solo",
                    "signal": "ema_mom",
                    "params": {
                        "ema_period": ema,
                        "sessions": list(SESSIONS),
                        "allow_short": True,
                    },
                    "stop_points": stop,
                }
            )
        for lb in (5, 10, 15, 20):
            combos.append(
                {
                    "kind": "solo",
                    "signal": "breakout",
                    "params": {"lookback": lb, "sessions": list(SESSIONS)},
                    "stop_points": stop,
                }
            )

    # TORB：允許一日多次突破
    for probe in (5, 10, 15, 30):
        for tp in (0.0, 30.0, 50.0):
            for stop in (150.0, 300.0):
                combos.append(
                    {
                        "kind": "solo",
                        "signal": "torb",
                        "params": {
                            "probe_minutes": probe,
                            "take_profit": tp,
                            "sessions": list(SESSIONS),
                            "allow_short": True,
                            "once_per_day": False,
                        },
                        "stop_points": stop,
                    }
                )

    # 多邊小停利（日夜）— 覆蓋未必高，當對照
    for period, low, tp, stop in (
        (8, 25.0, 30.0, 200.0),
        (10, 30.0, 40.0, 200.0),
        (14, 25.0, 40.0, 300.0),
        (21, 20.0, 40.0, 300.0),
    ):
        combos.append(
            {
                "kind": "solo",
                "signal": "rsi_tp",
                "params": {
                    "rsi_period": period,
                    "rsi_low": low,
                    "take_profit": tp,
                    "sessions": list(SESSIONS),
                    "allow_short": False,
                },
                "stop_points": stop,
            }
        )

    # 空邊小停利（日夜）
    for period, high, tp, stop in (
        (8, 75.0, 30.0, 200.0),
        (10, 70.0, 40.0, 200.0),
        (14, 75.0, 40.0, 300.0),
        (21, 80.0, 40.0, 300.0),
    ):
        combos.append(
            {
                "kind": "solo",
                "signal": "rsi_tp_short",
                "params": {
                    "rsi_period": period,
                    "rsi_high": high,
                    "take_profit": tp,
                    "sessions": list(SESSIONS),
                    "allow_short": True,
                },
                "stop_points": stop,
            }
        )

    return combos


def _run_solo(bars, spec: dict, trading: dict, slip: float):
    params = spec["params"]
    sessions = tuple(params.get("sessions") or SESSIONS)
    stop = float(spec["stop_points"])
    name = spec["signal"]

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

    allow_short = bool(params.get("allow_short", True))
    cfg = _cfg(trading, slip, stop=stop, sessions=sessions, allow_short=allow_short)
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
    if name == "torb":
        return run_torb_fast(
            bars,
            int(params["probe_minutes"]),
            cfg,
            take_profit_points=float(params.get("take_profit", 0.0)),
            once_per_day=bool(params.get("once_per_day", False)),
        )
    if name == "rsi_tp":
        return run_rsi_tp_fast(
            bars,
            int(params["rsi_period"]),
            float(params["rsi_low"]),
            cfg,
            take_profit_points=float(params["take_profit"]),
        )
    if name == "rsi_tp_short":
        return run_rsi_tp_short_fast(
            bars,
            int(params["rsi_period"]),
            float(params["rsi_high"]),
            cfg,
            take_profit_points=float(params["take_profit"]),
        )
    raise ValueError(name)


def _run_dual(bars, long_spec: dict, short_spec: dict, trading: dict, slip: float):
    lp, sp = long_spec["params"], short_spec["params"]
    cfg = _cfg(trading, slip, stop=0.0, sessions=SESSIONS, allow_short=True)
    return run_dual_rsi_tp_fast(
        bars,
        cfg,
        long_period=int(lp["rsi_period"]),
        long_low=float(lp["rsi_low"]),
        long_tp=float(lp["take_profit"]),
        long_stop=float(long_spec["stop_points"]),
        short_period=int(sp["rsi_period"]),
        short_high=float(sp["rsi_high"]),
        short_tp=float(sp["take_profit"]),
        short_stop=float(short_spec["stop_points"]),
        prefer="long",
    )


def _slim(row: dict) -> dict:
    skip = {"IS", "OOS", "tier_pass", "edge_ok", "pass", "min_coverage", "kind"}
    return {k: v for k, v in row.items() if k not in skip} | {
        "IS": row["IS"],
        "OOS": row["OOS"],
        "edge_ok": row["edge_ok"],
        "min_coverage": row["min_coverage"],
        "tier_pass": row.get("tier_pass"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark daily-trade track (vs tmf_v1 lock)")
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument("--min-daily-win-rate", type=float, default=0.55)
    p.add_argument("--coverage-tiers", default="0.5,0.7,0.9")
    p.add_argument("--top-k", type=int, default=6)
    args = p.parse_args()
    min_dwr = float(args.min_daily_win_rate)
    tiers = tuple(
        float(x.strip()) for x in str(args.coverage_tiers).split(",") if x.strip()
    ) or DEFAULT_TIERS
    top_k = max(1, int(args.top_k))

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    capital = float(trading["capital"])
    is_bars = load_1m_range(args.is_start, args.is_end)
    oos_bars = load_1m_range(args.oos_start, args.oos_end)
    is_avail = _available_dates(is_bars)
    oos_avail = _available_dates(oos_bars)

    grid = _solo_grid()
    report_path = ROOT / "outputs" / "daily_trade_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "n_combos": len(grid),
                "min_daily_win_rate": min_dwr,
                "coverage_tiers": list(tiers),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} daily-trade scan n={len(grid)} "
        f"dwr>{min_dwr} tiers={list(tiers)} sessions=day+night"
    )

    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(grid):
        is_s = _score(_run_solo(is_bars, spec, trading, slip), capital, is_avail)
        oos_s = _score(_run_solo(oos_bars, spec, trading, slip), capital, oos_avail)
        edge = _edge_ok(is_s, oos_s, min_dwr)
        rows.append(
            {
                **spec,
                "IS": is_s,
                "OOS": oos_s,
                "edge_ok": edge,
                "min_coverage": round(
                    min(is_s["day_coverage"], oos_s["day_coverage"]), 4
                ),
                "tier_pass": _pass_tiers(is_s, oos_s, min_dwr, tiers),
                "pass": any(_pass_tiers(is_s, oos_s, min_dwr, tiers).values()),
            }
        )
        if (i + 1) % 40 == 0 or i + 1 == len(grid):
            logger.info(f"{notify_prefix()} daily-trade solo {i+1}/{len(grid)}")

    rows.sort(
        key=lambda r: (
            r["edge_ok"],
            r["min_coverage"],
            r["OOS"]["avg_trips_per_trade_day"],
            r["OOS"]["net_pnl"],
        ),
        reverse=True,
    )

    # 多／空分訊號合成：取 rsi_tp / rsi_tp_short edge
    long_edge = [
        r for r in rows if r["edge_ok"] and r["signal"] == "rsi_tp"
    ][:top_k]
    short_edge = [
        r for r in rows if r["edge_ok"] and r["signal"] == "rsi_tp_short"
    ][:top_k]
    # 若沒 edge，退而求其次：覆蓋最高的各取 top_k 做實驗合成（仍要最終 edge 才算過關）
    if not long_edge:
        long_edge = sorted(
            [r for r in rows if r["signal"] == "rsi_tp"],
            key=lambda r: r["min_coverage"],
            reverse=True,
        )[:top_k]
    if not short_edge:
        short_edge = sorted(
            [r for r in rows if r["signal"] == "rsi_tp_short"],
            key=lambda r: r["min_coverage"],
            reverse=True,
        )[:top_k]

    pair_rows: list[dict[str, Any]] = []
    if long_edge and short_edge:
        total = len(long_edge) * len(short_edge)
        done = 0
        for L in long_edge:
            for S in short_edge:
                is_s = _score(
                    _run_dual(is_bars, L, S, trading, slip), capital, is_avail
                )
                oos_s = _score(
                    _run_dual(oos_bars, L, S, trading, slip), capital, oos_avail
                )
                edge = _edge_ok(is_s, oos_s, min_dwr)
                pair_rows.append(
                    {
                        "kind": "dual",
                        "signal": "dual_rsi_tp",
                        "long": {
                            "params": L["params"],
                            "stop_points": L["stop_points"],
                        },
                        "short": {
                            "params": S["params"],
                            "stop_points": S["stop_points"],
                        },
                        "IS": is_s,
                        "OOS": oos_s,
                        "edge_ok": edge,
                        "min_coverage": round(
                            min(is_s["day_coverage"], oos_s["day_coverage"]), 4
                        ),
                        "tier_pass": _pass_tiers(is_s, oos_s, min_dwr, tiers),
                        "pass": any(
                            _pass_tiers(is_s, oos_s, min_dwr, tiers).values()
                        ),
                    }
                )
                done += 1
                if done % 5 == 0 or done == total:
                    logger.info(f"{notify_prefix()} daily-trade dual {done}/{total}")

    pair_rows.sort(
        key=lambda r: (
            r["edge_ok"],
            r["min_coverage"],
            r["OOS"]["avg_trips_per_trade_day"],
            r["OOS"]["net_pnl"],
        ),
        reverse=True,
    )

    def by_tier(rs: list[dict]) -> dict[str, Any]:
        out = {}
        for t in tiers:
            wins = [r for r in rs if r.get("tier_pass", {}).get(str(t))]
            out[str(t)] = {"n_pass": len(wins), "best": wins[0] if wins else None}
        return out

    all_rows = rows + pair_rows
    n_pass = sum(1 for r in all_rows if r.get("pass"))
    best = next((r for r in all_rows if r.get("edge_ok")), all_rows[0] if all_rows else None)

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "track": "tmf_daily_v1_research",
        "vs": "tmf_v1_high_wr_lock_untouched",
        "goal": "daily_coverage_day_night_long_short_multi_trips",
        "constraints": {
            "max_lots": 1,
            "reentry_same_day": True,
            "prefilter": False,
            "session_win_rates": "split_day_night",
        },
        "min_daily_win_rate": min_dwr,
        "coverage_tiers": list(tiers),
        "pass_rule": (
            f"IS>0 OOS>0 session_dwr>{min_dwr} (day/night split) "
            "IS_trades>=40 OOS_trades>=20 + coverage tier"
        ),
        "n_solo": len(rows),
        "n_dual": len(pair_rows),
        "n_edge_solo": len([r for r in rows if r["edge_ok"]]),
        "n_edge_dual": len([r for r in pair_rows if r["edge_ok"]]),
        "n_pass_any_tier": n_pass,
        "available_days": {"IS": len(is_avail), "OOS": len(oos_avail)},
        "solo_by_tier": by_tier(rows),
        "dual_by_tier": by_tier(pair_rows),
        "best_edge": best,
        "top15_solo": rows[:15],
        "top10_dual": pair_rows[:10],
        "note": (
            "日頻專用；不過關不 enable strategies.tmf_daily_v1。"
            "現役 tmf_v1＋prefilter 維持高勝率少單對照。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} daily-trade done "
        f"solo_edge={payload['n_edge_solo']} dual_edge={payload['n_edge_dual']} "
        f"pass={n_pass} -> {report_path}"
    )
    if best and best.get("edge_ok"):
        logger.info(
            f"  BEST_EDGE {best.get('signal')} cov={best['min_coverage']} "
            f"avg_trips={best['OOS'].get('avg_trips_per_trade_day')} "
            f"IS={best['IS']['net_pnl']} OOS={best['OOS']['net_pnl']}"
        )
    return 0 if n_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
