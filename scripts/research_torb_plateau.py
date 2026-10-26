#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TORB probe 網格 + RSI 鎖策略參數高原 + 五門閘門／DSR。

結果 → outputs/torb_plateau_report.json
過關不自動改 tmf_v1 鎖；僅研究輸出。
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

from src.backtest.fast_signals import FastSignalConfig, run_rsi_tp_fast, run_torb_fast
from src.backtest.metrics import (
    daily_pnl_stats,
    deflated_sharpe_ratio,
    pair_round_trips,
    sharpe_from_pnls,
    summarize,
    t_stat,
)
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range


def _cfg(
    trading: dict,
    slip: float,
    *,
    stop: float,
    flatten: bool = True,
    allow_short: bool = False,
) -> FastSignalConfig:
    return FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=("day",),
        stop_points=float(stop),
        max_daily_loss=0.0,
        flatten_eod=flatten,
        allow_short=allow_short,
    )


def _score(result, capital: float) -> dict[str, Any]:
    trips = pair_round_trips(result.trades)
    s = summarize(result, capital=capital, label="x")
    d = daily_pnl_stats(trips)
    trip_pnls = [float(t["pnl"]) for t in trips]
    day_pnls = list(d.get("daily_pnls", {}).values())
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "win_rate": s["win_rate"],
        "n_trade_days": d["n_trade_days"],
        "n_win_days": d["n_win_days"],
        "n_loss_days": d["n_loss_days"],
        "daily_win_rate": d["daily_win_rate"],
        "max_drawdown": s["max_drawdown"],
        "t_stat_trips": t_stat(trip_pnls),
        "t_stat_days": t_stat(day_pnls),
        "sharpe_trips": sharpe_from_pnls(trip_pnls),
        "trip_pnls": trip_pnls,
        "day_pnls": day_pnls,
    }


def _five_gates(
    is_s: dict[str, Any],
    oos_s: dict[str, Any],
    *,
    min_oos_t: float = 2.0,
    min_oos_trades: int = 15,
    min_is_trades: int = 30,
) -> dict[str, Any]:
    """
    Adapted MNQ-style gates for tidemark IS/OOS split (costs already in engine):
    1) IS net > 0
    2) OOS net > 0
    3) OOS t-stat (trips) >= 2
    4) trade counts
    5) sign stability (IS & OOS both positive — same as 1+2; kept explicit)
    """
    oos_t = oos_s.get("t_stat_trips")
    gates = {
        "is_net_pos": float(is_s["net_pnl"]) > 0,
        "oos_net_pos": float(oos_s["net_pnl"]) > 0,
        "oos_tstat_ge_2": oos_t is not None and float(oos_t) >= min_oos_t,
        "trade_count": int(is_s["n_round_trips"]) >= min_is_trades
        and int(oos_s["n_round_trips"]) >= min_oos_trades,
        "sign_stable": float(is_s["net_pnl"]) > 0 and float(oos_s["net_pnl"]) > 0,
    }
    gates["pass_all"] = all(gates.values())
    gates["oos_t_stat"] = oos_t
    return gates


def _strip_pnls(score: dict[str, Any]) -> dict[str, Any]:
    out = dict(score)
    out.pop("trip_pnls", None)
    out.pop("day_pnls", None)
    # round floats for report
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 4) if abs(v) < 100 else round(v, 2)
    return out


def _plateau_score(rows: list[dict[str, Any]], center_id: str) -> float | None:
    """
    Neighbor similarity on OOS net_pnl for RSI plateau (higher = flatter island).
    Score = 1 / (1 + cv of neighbor OOS pnl), None if center missing.
    """
    by_id = {r["id"]: r for r in rows}
    center = by_id.get(center_id)
    if not center:
        return None
    cp = center["params"]
    neighbors: list[float] = []
    for r in rows:
        p = r["params"]
        dist = (
            abs(int(p["rsi_period"]) - int(cp["rsi_period"]))
            + abs(float(p["rsi_low"]) - float(cp["rsi_low"])) / 5.0
            + abs(float(p["take_profit_points"]) - float(cp["take_profit_points"])) / 10.0
            + abs(float(p["stop_points"]) - float(cp["stop_points"])) / 25.0
        )
        if 0 < dist <= 2.5:
            neighbors.append(float(r["OOS"]["net_pnl"]))
    if len(neighbors) < 3:
        return None
    mean = sum(neighbors) / len(neighbors)
    if mean == 0:
        return None
    var = sum((x - mean) ** 2 for x in neighbors) / len(neighbors)
    cv = math_sqrt(var) / abs(mean)
    return round(1.0 / (1.0 + cv), 4)


def math_sqrt(x: float) -> float:
    return float(x) ** 0.5


def main() -> int:
    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    capital = float(trading["capital"])
    is_bars = load_1m_range("2024-08-01", "2026-03-31")
    oos_bars = load_1m_range("2026-04-01", "2026-10-22")

    report_path = ROOT / "outputs" / "torb_plateau_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # --- TORB grid (long-only daytrade) ---
    probes = (5, 10, 15, 20, 25, 30, 35, 37, 40, 45, 50, 55, 60)
    tps = (0.0, 20.0, 30.0, 40.0, 50.0)
    stops = (0.0, 150.0, 200.0, 250.0, 300.0)
    torb_combos: list[dict[str, Any]] = []
    for probe in probes:
        for tp in tps:
            for stop in stops:
                if tp > 0 and stop > 0 and stop < tp * 2:
                    continue
                torb_combos.append({"probe": probe, "tp": tp, "stop": stop})

    # --- RSI plateau around lock ---
    rsi_combos: list[dict[str, Any]] = []
    for period in (26, 28, 30):
        for low in (18.0, 20.0, 22.0):
            for tp in (35.0, 40.0, 45.0):
                for stop in (275.0, 300.0, 325.0):
                    rsi_combos.append(
                        {
                            "rsi_period": period,
                            "rsi_low": low,
                            "take_profit_points": tp,
                            "stop_points": stop,
                        }
                    )

    n_trials = len(torb_combos) + len(rsi_combos)
    report_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "n_torb": len(torb_combos),
                "n_rsi_plateau": len(rsi_combos),
                "n_trials": n_trials,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} TORB+plateau scan torb={len(torb_combos)} "
        f"rsi={len(rsi_combos)} trials={n_trials}"
    )

    # baseline lock
    lock_cfg = _cfg(trading, slip, stop=300)
    rsi_is = run_rsi_tp_fast(is_bars, 28, 20.0, lock_cfg, take_profit_points=40.0)
    rsi_oos = run_rsi_tp_fast(oos_bars, 28, 20.0, lock_cfg, take_profit_points=40.0)
    base_is, base_oos = _score(rsi_is, capital), _score(rsi_oos, capital)
    baseline = {
        "id": "locked_rsi28_tp40_stop300",
        "signal": "rsi_tp",
        "params": {
            "rsi_period": 28,
            "rsi_low": 20,
            "take_profit_points": 40,
            "stop_points": 300,
        },
        "IS": _strip_pnls(base_is),
        "OOS": _strip_pnls(base_oos),
        "gates": _five_gates(base_is, base_oos),
        "dsr_oos": deflated_sharpe_ratio(
            base_oos.get("sharpe_trips"),
            n_trials=n_trials,
            n_obs=max(int(base_oos["n_round_trips"]), 2),
        ),
    }

    torb_rows: list[dict[str, Any]] = []
    for i, c in enumerate(torb_combos):
        cfg = _cfg(trading, slip, stop=c["stop"], allow_short=False)
        is_r = run_torb_fast(
            is_bars, c["probe"], cfg, take_profit_points=c["tp"], once_per_day=True
        )
        oos_r = run_torb_fast(
            oos_bars, c["probe"], cfg, take_profit_points=c["tp"], once_per_day=True
        )
        is_s, oos_s = _score(is_r, capital), _score(oos_r, capital)
        gates = _five_gates(is_s, oos_s)
        dsr = deflated_sharpe_ratio(
            oos_s.get("sharpe_trips"),
            n_trials=n_trials,
            n_obs=max(int(oos_s["n_round_trips"]), 2),
        )
        torb_rows.append(
            {
                "id": f"torb_p{c['probe']}_tp{int(c['tp'])}_stop{int(c['stop'])}",
                "signal": "torb",
                "params": c,
                "IS": _strip_pnls(is_s),
                "OOS": _strip_pnls(oos_s),
                "gates": gates,
                "dsr_oos": None if dsr is None else round(dsr, 4),
            }
        )
        if (i + 1) % 40 == 0 or i + 1 == len(torb_combos):
            logger.info(f"{notify_prefix()} TORB {i+1}/{len(torb_combos)}")

    rsi_rows: list[dict[str, Any]] = []
    for i, c in enumerate(rsi_combos):
        cfg = _cfg(trading, slip, stop=c["stop_points"])
        is_r = run_rsi_tp_fast(
            is_bars,
            c["rsi_period"],
            c["rsi_low"],
            cfg,
            take_profit_points=c["take_profit_points"],
        )
        oos_r = run_rsi_tp_fast(
            oos_bars,
            c["rsi_period"],
            c["rsi_low"],
            cfg,
            take_profit_points=c["take_profit_points"],
        )
        is_s, oos_s = _score(is_r, capital), _score(oos_r, capital)
        gates = _five_gates(is_s, oos_s)
        dsr = deflated_sharpe_ratio(
            oos_s.get("sharpe_trips"),
            n_trials=n_trials,
            n_obs=max(int(oos_s["n_round_trips"]), 2),
        )
        rid = (
            f"rsi{c['rsi_period']}_lo{int(c['rsi_low'])}"
            f"_tp{int(c['take_profit_points'])}_stop{int(c['stop_points'])}"
        )
        rsi_rows.append(
            {
                "id": rid,
                "signal": "rsi_tp",
                "params": c,
                "IS": _strip_pnls(is_s),
                "OOS": _strip_pnls(oos_s),
                "gates": gates,
                "dsr_oos": None if dsr is None else round(dsr, 4),
            }
        )
        if (i + 1) % 20 == 0 or i + 1 == len(rsi_combos):
            logger.info(f"{notify_prefix()} RSI plateau {i+1}/{len(rsi_combos)}")

    def _rank_key(r: dict[str, Any]) -> tuple:
        g = r["gates"]
        return (
            bool(g.get("pass_all")),
            float(r.get("dsr_oos") or 0),
            float(r["OOS"].get("t_stat_trips") or 0),
            float(r["OOS"]["net_pnl"]),
            float(r["IS"]["net_pnl"]),
        )

    torb_rows.sort(key=_rank_key, reverse=True)
    rsi_rows.sort(key=_rank_key, reverse=True)
    torb_pass = [r for r in torb_rows if r["gates"]["pass_all"]]
    rsi_pass = [r for r in rsi_rows if r["gates"]["pass_all"]]

    lock_plateau = _plateau_score(rsi_rows, "rsi28_lo20_tp40_stop300")

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "pass_rule": (
            "five_gates: IS>0 OOS>0 OOS_t_stat_trips>=2 "
            "IS_trades>=30 OOS_trades>=15 sign_stable; DSR uses n_trials"
        ),
        "ranges": {
            "IS": ["2024-08-01", "2026-03-31"],
            "OOS": ["2026-04-01", "2026-10-22"],
        },
        "n_trials": n_trials,
        "n_torb": len(torb_combos),
        "n_rsi_plateau": len(rsi_combos),
        "n_torb_pass": len(torb_pass),
        "n_rsi_pass": len(rsi_pass),
        "baseline_rsi_lock": baseline,
        "lock_plateau_score": lock_plateau,
        "best_torb": torb_rows[0] if torb_rows else None,
        "best_rsi_plateau": rsi_rows[0] if rsi_rows else None,
        "torb_winners": torb_pass[:15],
        "rsi_winners": rsi_pass[:15],
        "torb_top10": torb_rows[:10],
        "rsi_top10": rsi_rows[:10],
        "note": (
            "TORB=日盤開盤區間突破（probe 分）；RSI 高原=鎖策略鄰域。"
            "過關不改 tmf_v1；對照五門閘門與 DSR。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} TORB+plateau done "
        f"torb_pass={len(torb_pass)}/{len(torb_combos)} "
        f"rsi_pass={len(rsi_pass)}/{len(rsi_combos)} -> {report_path}"
    )
    if torb_rows:
        b = torb_rows[0]
        logger.info(
            f"  BEST_TORB {b['params']} pass={b['gates']['pass_all']} "
            f"IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']} "
            f"t={b['OOS'].get('t_stat_trips')} dsr={b.get('dsr_oos')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
