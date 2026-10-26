#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
秒級複核鎖策略：1m 訊號 + 1s 停利／停損成交。

政策（接受 SIM tick 洞）：
- 覆蓋率不足時，1s 結果僅作 REFERENCE，不挡主閘門（主證據=全歷史 1m + paper）。
- 覆蓋率達標才啟用 STRICT（IS&OOS path 皆過才算 pass）。

結果 → outputs/verify_1s_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.fast_signals import FastSignalConfig, run_rsi_tp_fast
from src.backtest.metrics import summarize
from src.backtest.verify_1s import run_rsi_tp_1m_signal_1s_fill
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range, load_1s_range
from src.data.local_store import LocalStore

# 低於此覆蓋率 → 1s 不挡主流程（SIM tick 洞）
COVERAGE_STRICT_MIN = 0.50


def _nonempty_1s_dates(store: LocalStore, start: str, end: str) -> list[str]:
    out: list[str] = []
    for d in store.list_dates("TMF", "1s"):
        if d < start or d > end:
            continue
        df = store.load("TMF", d, "1s")
        if df is not None and not df.empty:
            out.append(d)
    return out


def _nonempty_1m_dates(store: LocalStore, start: str, end: str) -> list[str]:
    out: list[str] = []
    for d in store.list_dates("TMF", "1m"):
        if d < start or d > end:
            continue
        df = store.load("TMF", d, "1m")
        if df is not None and not df.empty:
            out.append(d)
    return out


def _filter_days(bars: pd.DataFrame, days: set[str]) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    mask = df["ts"].dt.strftime("%Y-%m-%d").isin(days)
    return df.loc[mask].reset_index(drop=True)


def _pack(label: str, result, capital: float) -> dict:
    s = summarize(result, capital=capital, label=label)
    return {
        "label": label,
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "win_rate": s["win_rate"],
        "n_trade_days": s["n_trade_days"],
        "n_win_days": s["n_win_days"],
        "daily_win_rate": s["daily_win_rate"],
        "max_drawdown": s["max_drawdown"],
        "final_equity": s["final_equity"],
        "n_steps": s["n_steps"],
    }


def _cfg(trading: dict, slip: float, params: dict) -> FastSignalConfig:
    return FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=tuple(params.get("sessions") or ["day"]),
        stop_points=float(params.get("stop_points") or 0),
        max_daily_loss=float(params.get("max_daily_loss") or 0),
        flatten_eod=bool(params.get("flatten_eod", True)),
        allow_short=bool(params.get("allow_short", False)),
    )


def _ok(hyb: dict) -> bool:
    return bool(
        (hyb.get("daily_win_rate") or 0) > 0.8
        and hyb["net_pnl"] > 0
        and (hyb.get("n_trade_days") or 0) >= 5
    )


def _coverage(n_1s: int, n_1m: int) -> dict:
    cov = round(n_1s / n_1m, 4) if n_1m > 0 else 0.0
    return {
        "n_1m_days": n_1m,
        "n_1s_days": n_1s,
        "coverage": cov,
        "strict_eligible": cov >= COVERAGE_STRICT_MIN,
    }


def _eval_split(
    name: str,
    start: str,
    end: str,
    days: list[str],
    trading: dict,
    slip: float,
    params: dict,
) -> dict:
    day_set = set(days)
    bars_1m = _filter_days(load_1m_range(start, end), day_set)
    bars_1s = _filter_days(load_1s_range(start, end), day_set)
    cfg = _cfg(trading, slip, params)
    period = int(params.get("rsi_period", 28))
    lo = float(params.get("rsi_low", 20))
    tp = float(params.get("take_profit_points", 40))
    cap = float(trading["capital"])

    base = run_rsi_tp_fast(bars_1m, period, lo, cfg, take_profit_points=tp)
    raw = run_rsi_tp_1m_signal_1s_fill(
        bars_1m,
        bars_1s,
        period,
        lo,
        cfg,
        take_profit_points=tp,
        min_hold_seconds=0,
        entry_lag_minutes=0,
    )
    path = run_rsi_tp_1m_signal_1s_fill(
        bars_1m,
        bars_1s,
        period,
        lo,
        cfg,
        take_profit_points=tp,
        min_hold_seconds=60,
        entry_lag_minutes=1,
    )
    base_s = _pack(f"{name}_1m", base, cap)
    raw_s = _pack(f"{name}_1s_raw", raw, cap)
    path_s = _pack(f"{name}_1s_path", path, cap)
    return {
        "range": [start, end],
        "n_days_1s": len(days),
        "n_bars_1m": len(bars_1m),
        "n_bars_1s": len(bars_1s),
        "baseline_1m": base_s,
        "hybrid_1s_fill": raw_s,
        "hybrid_1s_path": path_s,
        "pnl_delta_raw": round(raw_s["net_pnl"] - base_s["net_pnl"], 2),
        "pnl_delta_path": round(path_s["net_pnl"] - base_s["net_pnl"], 2),
        "daily_wr_ok_raw": _ok(raw_s),
        "daily_wr_ok_path": _ok(path_s),
        "daily_wr_ok": _ok(path_s),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="tidemark 1s verify locked rsi_tp")
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument(
        "--coverage-min",
        type=float,
        default=COVERAGE_STRICT_MIN,
        help="IS&OOS 覆蓋率皆達標才啟用 STRICT 閘門",
    )
    args = p.parse_args()
    cov_min = float(args.coverage_min)

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    strat = next(
        (s for s in settings.get("strategies", []) if s.get("id") == "tmf_v1"),
        {},
    )
    params = dict(strat.get("params") or {})
    store = LocalStore(settings)
    report_path = ROOT / "outputs" / "verify_1s_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"status": "RUNNING", "updated_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )

    is_1m = _nonempty_1m_dates(store, args.is_start, args.is_end)
    oos_1m = _nonempty_1m_dates(store, args.oos_start, args.oos_end)
    is_days = _nonempty_1s_dates(store, args.is_start, args.is_end)
    oos_days = _nonempty_1s_dates(store, args.oos_start, args.oos_end)
    cov_is = _coverage(len(is_days), len(is_1m))
    cov_oos = _coverage(len(oos_days), len(oos_1m))
    # recompute strict_eligible with CLI threshold
    cov_is["strict_eligible"] = cov_is["coverage"] >= cov_min
    cov_oos["strict_eligible"] = cov_oos["coverage"] >= cov_min
    strict_eligible = bool(cov_is["strict_eligible"] and cov_oos["strict_eligible"])

    logger.info(
        f"{notify_prefix()} 1s verify IS={len(is_days)}/{len(is_1m)} "
        f"OOS={len(oos_days)}/{len(oos_1m)} strict_eligible={strict_eligible}"
    )

    is_block = _eval_split(
        "IS", args.is_start, args.is_end, is_days, trading, slip, params
    )
    oos_block = _eval_split(
        "OOS", args.oos_start, args.oos_end, oos_days, trading, slip, params
    )
    is_block["coverage"] = cov_is
    oos_block["coverage"] = cov_oos

    pass_raw = bool(is_block["daily_wr_ok_raw"] and oos_block["daily_wr_ok_raw"])
    pass_path = bool(is_block["daily_wr_ok_path"] and oos_block["daily_wr_ok_path"])
    pass_oos_ref = bool(oos_block["daily_wr_ok_path"])

    if strict_eligible:
        gate = "STRICT_PASS" if pass_path else "STRICT_FAIL"
    else:
        gate = "REFERENCE_ONLY"

    # 主流程：覆蓋不足時不因 1s 失敗而挡；STRICT 才用 path 結果
    blocking_fail = gate == "STRICT_FAIL"
    payload = {
        "status": "DONE",
        "brand": "tidemark",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "1m_signal_1s_fill",
        "policy": {
            "name": "accept_sim_tick_gaps",
            "coverage_min_for_strict": cov_min,
            "primary_evidence": ["full_history_1m_IS_OOS", "paper_replay"],
            "one_s_role": (
                "reference_when_coverage_low"
                if not strict_eligible
                else "strict_gate"
            ),
        },
        "path_rules": {
            "raw": {"min_hold_seconds": 0, "entry_lag_minutes": 0},
            "path": {"min_hold_seconds": 60, "entry_lag_minutes": 1},
        },
        "params": {
            k: params.get(k)
            for k in (
                "rsi_period",
                "rsi_low",
                "take_profit_points",
                "stop_points",
                "sessions",
                "flatten_eod",
                "max_daily_loss",
                "allow_short",
            )
        },
        "coverage": {"IS": cov_is, "OOS": cov_oos, "strict_eligible": strict_eligible},
        "IS": is_block,
        "OOS": oos_block,
        "pass_1s_raw": pass_raw,
        "pass_1s_path": pass_path,
        "pass_1s_oos_subset": pass_oos_ref,
        "pass_1s_strict": pass_path if strict_eligible else None,
        "gate_1s": gate,
        "pass_1s": (not blocking_fail) and (gate != "STRICT_FAIL"),
        "note": (
            f"覆蓋率<{cov_min} → gate=REFERENCE_ONLY（1s 不挡主流程；主證據=1m+paper）。"
            f"覆蓋達標 → STRICT（IS&OOS path 淨利>0 且日勝率>0.8）。"
            "raw=訊號分末秒進場；path=下一分首秒+60s持倉。"
        ),
    }
    # pass_1s meaning under new policy:
    # REFERENCE_ONLY → True (accepted limits); STRICT_PASS → True; STRICT_FAIL → False
    payload["pass_1s"] = gate in ("REFERENCE_ONLY", "STRICT_PASS")

    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} 1s verify gate={gate} pass_1s={payload['pass_1s']} "
        f"strict_path={pass_path} oos_ref={pass_oos_ref} -> {report_path}"
    )
    return 0 if payload["pass_1s"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
