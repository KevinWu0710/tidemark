#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多訊號／空訊號分開研究，再合成（一口、日+夜）。

1) 單獨掃做多族（rsi_tp）→ edge（賺 + 日勝率>門檻）
2) 單獨掃做空族（rsi_tp_short）→ edge
3) edge 候選兩兩合成（獨立進場 mask，max 等效 1 口）

不過關不改 tmf_v1。
結果 → outputs/long_short_split_report.json
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

from src.backtest.fast_signals import (
    FastSignalConfig,
    run_dual_rsi_tp_fast,
    run_rsi_tp_fast,
    run_rsi_tp_short_fast,
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
    allow_short: bool,
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
        "daily_win_rate": s.get("day_daily_win_rate"),
        "night_daily_win_rate": s.get("night_daily_win_rate"),
        "by_session": s.get("by_session"),
        "max_drawdown": s["max_drawdown"],
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
        and is_s["n_round_trips"] >= 20
        and oos_s["n_round_trips"] >= 10
    )


def _pass_tiers(is_s: dict, oos_s: dict, min_dwr: float, tiers: tuple[float, ...]) -> dict[str, bool]:
    edge = _edge_ok(is_s, oos_s, min_dwr)
    out: dict[str, bool] = {}
    for t in tiers:
        out[str(t)] = bool(
            edge
            and is_s["day_coverage"] >= t
            and oos_s["day_coverage"] >= t
        )
    return out


def _long_grid() -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for sessions in (["day"], list(SESSIONS_DN)):
        for period in (21, 28, 35):
            for low in (15.0, 20.0, 25.0):
                for tp in (30.0, 40.0, 50.0, 60.0):
                    for stop in (200.0, 300.0):
                        combos.append(
                            {
                                "side": "long",
                                "signal": "rsi_tp",
                                "params": {
                                    "rsi_period": period,
                                    "rsi_low": low,
                                    "take_profit": tp,
                                    "sessions": sessions,
                                },
                                "stop_points": stop,
                            }
                        )
    return combos


def _short_grid() -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for sessions in (["day"], list(SESSIONS_DN)):
        for period in (21, 28, 35):
            for high in (70.0, 75.0, 80.0, 85.0):
                for tp in (30.0, 40.0, 50.0, 60.0):
                    for stop in (200.0, 300.0):
                        combos.append(
                            {
                                "side": "short",
                                "signal": "rsi_tp_short",
                                "params": {
                                    "rsi_period": period,
                                    "rsi_high": high,
                                    "take_profit": tp,
                                    "sessions": sessions,
                                },
                                "stop_points": stop,
                            }
                        )
    return combos


def _run_solo(bars, spec: dict, trading: dict, slip: float):
    params = spec["params"]
    sessions = tuple(params["sessions"])
    stop = float(spec["stop_points"])
    if spec["side"] == "long":
        cfg = _cfg(trading, slip, stop=stop, sessions=sessions, allow_short=False)
        return run_rsi_tp_fast(
            bars,
            int(params["rsi_period"]),
            float(params["rsi_low"]),
            cfg,
            take_profit_points=float(params["take_profit"]),
        )
    cfg = _cfg(trading, slip, stop=stop, sessions=sessions, allow_short=True)
    return run_rsi_tp_short_fast(
        bars,
        int(params["rsi_period"]),
        float(params["rsi_high"]),
        cfg,
        take_profit_points=float(params["take_profit"]),
    )


def _session_union(a: list[str], b: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for s in ("day", "night"):
        if s in a or s in b:
            out.append(s)
    return tuple(out) if out else SESSIONS_DN


def _run_pair(
    bars,
    long_spec: dict,
    short_spec: dict,
    trading: dict,
    slip: float,
    prefer: str,
):
    lp = long_spec["params"]
    sp = short_spec["params"]
    sessions = _session_union(list(lp["sessions"]), list(sp["sessions"]))
    cfg = _cfg(trading, slip, stop=0.0, sessions=sessions, allow_short=True)
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
        prefer=prefer,
    )


def _slim(row: dict) -> dict:
    return {
        "side": row.get("side"),
        "signal": row["signal"],
        "params": row["params"],
        "stop_points": row["stop_points"],
        "IS": row["IS"],
        "OOS": row["OOS"],
        "edge_ok": row["edge_ok"],
        "min_coverage": row["min_coverage"],
        "tier_pass": row.get("tier_pass"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark long/short split then combine")
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument("--min-daily-win-rate", type=float, default=0.7)
    p.add_argument("--coverage-tiers", default="0.15,0.3,0.5,0.8")
    p.add_argument("--top-k", type=int, default=8, help="各邊取 edge 前 K 做兩兩合成")
    p.add_argument("--prefer", choices=("long", "short"), default="long")
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

    long_grid = _long_grid()
    short_grid = _short_grid()
    n_total = len(long_grid) + len(short_grid) + top_k * top_k
    report_path = ROOT / "outputs" / "long_short_split_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "n_long": len(long_grid),
                "n_short": len(short_grid),
                "top_k": top_k,
                "min_daily_win_rate": min_dwr,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} long/short-split start "
        f"long={len(long_grid)} short={len(short_grid)} top_k={top_k} dwr>{min_dwr}"
    )

    def scan_side(grid: list[dict], label: str) -> list[dict]:
        rows: list[dict] = []
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
                }
            )
            if (i + 1) % 40 == 0 or i + 1 == len(grid):
                logger.info(f"{notify_prefix()} {label} {i+1}/{len(grid)}")
        rows.sort(
            key=lambda r: (
                r["edge_ok"],
                r["min_coverage"],
                r["OOS"]["net_pnl"],
                r["IS"]["net_pnl"],
            ),
            reverse=True,
        )
        return rows

    long_rows = scan_side(long_grid, "long-solo")
    short_rows = scan_side(short_grid, "short-solo")
    long_edge = [r for r in long_rows if r["edge_ok"]][:top_k]
    short_edge = [r for r in short_rows if r["edge_ok"]][:top_k]

    logger.info(
        f"{notify_prefix()} solo edge long={len([r for r in long_rows if r['edge_ok']])} "
        f"short={len([r for r in short_rows if r['edge_ok']])} "
        f"pair={len(long_edge)}x{len(short_edge)}"
    )

    pair_rows: list[dict] = []
    if long_edge and short_edge:
        done = 0
        total_pairs = len(long_edge) * len(short_edge)
        for L in long_edge:
            for S in short_edge:
                is_s = _score(
                    _run_pair(is_bars, L, S, trading, slip, args.prefer),
                    capital,
                    is_avail,
                )
                oos_s = _score(
                    _run_pair(oos_bars, L, S, trading, slip, args.prefer),
                    capital,
                    oos_avail,
                )
                edge = _edge_ok(is_s, oos_s, min_dwr)
                pair_rows.append(
                    {
                        "signal": "dual_rsi_tp",
                        "prefer": args.prefer,
                        "long": _slim(L),
                        "short": _slim(S),
                        "sessions": list(
                            _session_union(
                                list(L["params"]["sessions"]),
                                list(S["params"]["sessions"]),
                            )
                        ),
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
                done += 1
                if done % 10 == 0 or done == total_pairs:
                    logger.info(f"{notify_prefix()} pair {done}/{total_pairs}")

    pair_rows.sort(
        key=lambda r: (
            r["edge_ok"],
            r["min_coverage"],
            r["OOS"]["net_pnl"],
            r["IS"]["net_pnl"],
        ),
        reverse=True,
    )

    def by_tier(rows: list[dict]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for t in tiers:
            key = str(t)
            wins = [r for r in rows if r.get("tier_pass", {}).get(key)]
            out[key] = {
                "n_pass": len(wins),
                "best": wins[0] if wins else None,
            }
        return out

    pair_edge = [r for r in pair_rows if r["edge_ok"]]
    any_pass = any(
        r.get("tier_pass", {}).get(str(t)) for r in pair_rows for t in tiers
    ) or any(
        r.get("tier_pass", {}).get(str(t)) for r in long_rows + short_rows for t in tiers
    )

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "goal": "separate_long_short_signals_then_combine_max1",
        "min_daily_win_rate": min_dwr,
        "coverage_tiers": list(tiers),
        "prefer_on_conflict": args.prefer,
        "pass_rule": (
            f"IS>0 OOS>0 daily_win_rate>{min_dwr} trades門檻；"
            "各 tier 另要求 day_coverage>=tier"
        ),
        "n_long_combos": len(long_grid),
        "n_short_combos": len(short_grid),
        "n_pairs": len(pair_rows),
        "n_long_edge": len([r for r in long_rows if r["edge_ok"]]),
        "n_short_edge": len([r for r in short_rows if r["edge_ok"]]),
        "n_pair_edge": len(pair_edge),
        "available_days": {"IS": len(is_avail), "OOS": len(oos_avail)},
        "long_solo": {
            "by_tier": by_tier(long_rows),
            "best_edge": _slim(long_edge[0]) if long_edge else None,
            "top5_edge": [_slim(r) for r in long_edge[:5]],
        },
        "short_solo": {
            "by_tier": by_tier(short_rows),
            "best_edge": _slim(short_edge[0]) if short_edge else None,
            "top5_edge": [_slim(r) for r in short_edge[:5]],
        },
        "combined": {
            "by_tier": by_tier(pair_rows),
            "best_edge": pair_edge[0] if pair_edge else None,
            "top10": pair_rows[:10],
        },
        "note": (
            "多／空為不同訊號；合成時一口、有倉不反向開。"
            "合並過關才考慮改鎖；否則維持 tmf_v1。"
        ),
        "approx_work_units": n_total,
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} long/short-split done "
        f"long_edge={payload['n_long_edge']} short_edge={payload['n_short_edge']} "
        f"pair_edge={payload['n_pair_edge']} -> {report_path}"
    )
    if pair_edge:
        b = pair_edge[0]
        logger.info(
            f"  BEST_PAIR cov={b['min_coverage']} "
            f"dwr={b['IS']['daily_win_rate']}/{b['OOS']['daily_win_rate']} "
            f"IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']}"
        )
    elif long_edge:
        b = long_edge[0]
        logger.info(
            f"  BEST_LONG_ONLY cov={b['min_coverage']} "
            f"IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']}"
        )
    return 0 if (pair_edge or any_pass) else 2


if __name__ == "__main__":
    raise SystemExit(main())
