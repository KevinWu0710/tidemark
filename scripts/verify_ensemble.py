#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
驗證 settings.yaml 雙軌（tmf_daily_secure / tmf_daily_trend）：
1. 研究引擎（HL 觸價，與 research_orb_advanced 同源）重現宣稱數字
2. 正式引擎近似（收盤觸價，對齊 TmfSessionStrategy._check_stop_or_tp）
3. 進場日重疊：證明是否「互補」或「子集」

結果：outputs/ensemble_verify_report.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

from src.backtest.metrics import pair_round_trips, summarize  # noqa: E402
from src.core.config import load_settings, notify_prefix  # noqa: E402
from src.core.types import Side  # noqa: E402
from src.data.bars import load_1m_range  # noqa: E402


def _load_orb_advanced():
    path = ROOT / "scripts" / "research_orb_advanced.py"
    spec = importlib.util.spec_from_file_location("research_orb_advanced", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclass 需要模組已註冊
    spec.loader.exec_module(mod)
    return mod


_orb = _load_orb_advanced()
TradeConfig = _orb.TradeConfig
evaluate_advanced_splits = _orb.evaluate_advanced_splits
prepare_advanced_features = _orb.prepare_advanced_features
run_advanced_simulation = _orb.run_advanced_simulation

REPORT_PATH = ROOT / "outputs" / "ensemble_verify_report.json"
IS_END = "2026-03-31"
OOS_START = "2026-04-01"
FULL_START = "2024-07-29"
FULL_END = "2026-10-23"

# 文件宣稱（STRATEGY_ENSEMBLE.md）
CLAIMED = {
    "tmf_daily_secure": {"net_pnl": 15411, "win_rate": 0.599, "oos_win_rate": 0.632},
    "tmf_daily_trend": {"net_pnl": 30801, "win_rate": 0.443, "oos_win_rate": 0.421},
}


def _strat_params(settings: dict[str, Any], sid: str) -> dict[str, Any]:
    cfg = next(s for s in settings["strategies"] if s["id"] == sid)
    p = dict(cfg["params"])
    return {
        "probe_mins": int(p.get("probe_mins", 15)),
        "cutoff_mins": int(p.get("cutoff_mins", 105)),
        "gap_th": float(p.get("gap_th", 40)),
        "max_or_range": float(p.get("max_or_range", 250)),
        "stop_points": float(p.get("stop_points", 50)),
        "tp_points": float(p.get("take_profit_points", 200)),
        "be_trig_pts": float(p.get("breakeven_points", 0)),
        "be_lock_pts": 5.0,
        "trail_trig_pts": 0.0,
        "trail_dist": 0.0,
        "time_stop_mins": int(p.get("time_stop_bars", 45)),
        "use_trend_filter": False,
        "vol_spike_mult": float(p.get("vol_spike_mult", 0)),
        "allow_short": bool(p.get("allow_short", False)),
    }


def run_close_based_simulation(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
    cfg: TradeConfig,
    *,
    probe_mins: int = 15,
    cutoff_mins: int = 105,
    gap_th: float = 40.0,
    max_or_range: float = 250.0,
    stop_points: float = 50.0,
    tp_points: float = 200.0,
    be_trig_pts: float = 0.0,
    be_lock_pts: float = 5.0,
    trail_trig_pts: float = 0.0,  # unused; keep signature parity
    trail_dist: float = 0.0,
    time_stop_mins: int = 45,
    use_trend_filter: bool = False,
    vol_spike_mult: float = 0.0,
    allow_short: bool = False,
):
    """對齊正式策略：停損／停利／保本用收盤價，不用棒內 high/low 觸價。"""
    from src.backtest.engine import BacktestResult

    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    volume = df["volume"].astype(float).values
    ts = df["ts"].values

    equity = cfg.capital
    trades: list[dict[str, Any]] = []

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    def fill_buy(mid: float) -> float:
        return mid + cfg.slip_ticks

    def fill_sell(mid: float) -> float:
        return mid - cfg.slip_ticks

    for sess in day_sessions:
        s_idx = sess["start_idx"]
        e_idx = sess["end_idx"]
        n_bars = e_idx - s_idx
        if n_bars <= probe_mins + 2:
            continue

        probe_high = float(np.max(high[s_idx : s_idx + probe_mins]))
        probe_low = float(np.min(low[s_idx : s_idx + probe_mins]))
        or_range = probe_high - probe_low
        avg_probe_vol = float(np.mean(volume[s_idx : s_idx + probe_mins]))
        gap = sess["gap_pts"]
        pclose = sess["prev_day_close"]
        trend = sess["trend_ema"]
        d_open = sess["open_px"]

        if max_or_range > 0 and or_range > max_or_range:
            continue

        allow_l = (gap >= gap_th) and (np.isnan(pclose) or probe_low > pclose)
        allow_s = allow_short and (gap <= -gap_th) and (np.isnan(pclose) or probe_high < pclose)

        if use_trend_filter:
            if allow_l and d_open < trend:
                allow_l = False
            if allow_s and d_open > trend:
                allow_s = False

        if not allow_l and not allow_s:
            continue

        max_search_idx = min(e_idx, s_idx + cutoff_mins)
        pos = Side.FLAT
        entry_px = 0.0
        peak_px = 0.0
        valley_px = 0.0
        bars_held = 0

        for k in range(s_idx + probe_mins, e_idx):
            cur_c = float(close[k])
            cur_h = float(high[k])
            cur_l = float(low[k])
            cur_v = float(volume[k])
            cur_ts = ts[k]
            is_eod = k == e_idx - 1

            if pos == Side.LONG:
                bars_held += 1
                peak_px = max(peak_px, cur_h)
                unrealized_gain = peak_px - entry_px
                cur_sl = entry_px - stop_points
                if be_trig_pts > 0 and unrealized_gain >= be_trig_pts:
                    cur_sl = max(cur_sl, entry_px + be_lock_pts)
                tp_target = entry_px + tp_points

                # 收盤觸價（對齊 TmfSessionStrategy）
                if cur_c <= cur_sl:
                    f = fill_sell(cur_c)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    reason = "breakeven" if cur_sl >= entry_px else "stop_loss"
                    trades.append(
                        {"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": reason}
                    )
                    pos = Side.FLAT
                    break
                if cur_c >= tp_target:
                    f = fill_sell(cur_c)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    trades.append(
                        {
                            "ts": cur_ts,
                            "action": "CLOSE",
                            "price": f,
                            "pnl": net,
                            "reason": "take_profit",
                        }
                    )
                    pos = Side.FLAT
                    break
                if time_stop_mins > 0 and bars_held >= time_stop_mins:
                    f = fill_sell(cur_c)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    trades.append(
                        {
                            "ts": cur_ts,
                            "action": "CLOSE",
                            "price": f,
                            "pnl": net,
                            "reason": f"time_stop_{time_stop_mins}",
                        }
                    )
                    pos = Side.FLAT
                    break
                if is_eod:
                    f = fill_sell(cur_c)
                    net = (f - entry_px) * cfg.point_value - cost(f)
                    equity += net
                    trades.append(
                        {
                            "ts": cur_ts,
                            "action": "CLOSE",
                            "price": f,
                            "pnl": net,
                            "reason": "flatten_eod",
                        }
                    )
                    pos = Side.FLAT
                    break

            elif pos == Side.SHORT:
                bars_held += 1
                valley_px = min(valley_px, cur_l)
                unrealized_gain = entry_px - valley_px
                cur_sl = entry_px + stop_points
                if be_trig_pts > 0 and unrealized_gain >= be_trig_pts:
                    cur_sl = min(cur_sl, entry_px - be_lock_pts)
                tp_target = entry_px - tp_points

                if cur_c >= cur_sl:
                    f = fill_buy(cur_c)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    reason = "breakeven" if cur_sl <= entry_px else "stop_loss"
                    trades.append(
                        {"ts": cur_ts, "action": "CLOSE", "price": f, "pnl": net, "reason": reason}
                    )
                    pos = Side.FLAT
                    break
                if cur_c <= tp_target:
                    f = fill_buy(cur_c)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    trades.append(
                        {
                            "ts": cur_ts,
                            "action": "CLOSE",
                            "price": f,
                            "pnl": net,
                            "reason": "take_profit",
                        }
                    )
                    pos = Side.FLAT
                    break
                if time_stop_mins > 0 and bars_held >= time_stop_mins:
                    f = fill_buy(cur_c)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    trades.append(
                        {
                            "ts": cur_ts,
                            "action": "CLOSE",
                            "price": f,
                            "pnl": net,
                            "reason": f"time_stop_{time_stop_mins}",
                        }
                    )
                    pos = Side.FLAT
                    break
                if is_eod:
                    f = fill_buy(cur_c)
                    net = (entry_px - f) * cfg.point_value - cost(f)
                    equity += net
                    trades.append(
                        {
                            "ts": cur_ts,
                            "action": "CLOSE",
                            "price": f,
                            "pnl": net,
                            "reason": "flatten_eod",
                        }
                    )
                    pos = Side.FLAT
                    break

            elif pos == Side.FLAT and k < max_search_idx:
                vol_ok = True
                if vol_spike_mult > 0:
                    vol_ok = cur_v >= vol_spike_mult * avg_probe_vol
                if allow_l and cur_c > probe_high and vol_ok:
                    f = fill_buy(cur_c)
                    equity -= cost(f)
                    entry_px = f
                    peak_px = cur_c
                    bars_held = 0
                    pos = Side.LONG
                    trades.append(
                        {"ts": cur_ts, "action": "BUY", "price": f, "reason": "orb_break_high"}
                    )
                elif allow_s and cur_c < probe_low and vol_ok:
                    f = fill_sell(cur_c)
                    equity -= cost(f)
                    entry_px = f
                    valley_px = cur_c
                    bars_held = 0
                    pos = Side.SHORT
                    trades.append(
                        {"ts": cur_ts, "action": "SELL", "price": f, "reason": "orb_break_low"}
                    )

    return BacktestResult(
        equity_curve=[],
        trades=trades,
        final_equity=equity,
        n_steps=len(trades),
    )


def evaluate_close_splits(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
    cfg: TradeConfig,
    params: dict[str, Any],
) -> dict[str, Any]:
    is_sessions = [s for s in day_sessions if s["date"] <= IS_END]
    oos_sessions = [s for s in day_sessions if s["date"] >= OOS_START]
    res_is = run_close_based_simulation(df, is_sessions, cfg, **params)
    res_oos = run_close_based_simulation(df, oos_sessions, cfg, **params)
    trips_is = pair_round_trips(res_is.trades)
    trips_oos = pair_round_trips(res_oos.trades)
    summ_is = summarize(res_is, capital=cfg.capital, label="IS")
    summ_oos = summarize(res_oos, capital=cfg.capital, label="OOS")
    net_is = float(summ_is.get("net_pnl", 0.0))
    net_oos = float(summ_oos.get("net_pnl", 0.0))
    wr_is = float(summ_is.get("win_rate") or 0.0)
    wr_oos = float(summ_oos.get("win_rate") or 0.0)
    full_trips = trips_is + trips_oos
    full_wins = sum(1 for t in full_trips if t.get("pnl", 0.0) > 0)
    full_wr = (full_wins / len(full_trips)) if full_trips else 0.0
    return {
        "params": params,
        "pass": net_is > 0 and net_oos > 0 and len(trips_is) >= 20 and len(trips_oos) >= 10,
        "IS": {
            "net_pnl": round(net_is, 2),
            "trades": len(trips_is),
            "win_rate": round(wr_is, 4),
            "max_drawdown": round(float(summ_is.get("max_drawdown") or 0), 2),
            "daily_win_rate": summ_is.get("daily_win_rate"),
        },
        "OOS": {
            "net_pnl": round(net_oos, 2),
            "trades": len(trips_oos),
            "win_rate": round(wr_oos, 4),
            "max_drawdown": round(float(summ_oos.get("max_drawdown") or 0), 2),
            "daily_win_rate": summ_oos.get("daily_win_rate"),
        },
        "total_pnl": round(net_is + net_oos, 2),
        "total_trades": len(full_trips),
        "total_win_rate": round(full_wr, 4),
    }


def entry_dates(
    df: pd.DataFrame,
    day_sessions: list[dict[str, Any]],
    cfg: TradeConfig,
    params: dict[str, Any],
) -> set[str]:
    res = run_advanced_simulation(df, day_sessions, cfg, **params)
    out: set[str] = set()
    for t in res.trades:
        if t.get("action") in ("BUY", "SELL"):
            ts = pd.Timestamp(t["ts"])
            out.add(ts.strftime("%Y-%m-%d"))
    return out


def main() -> int:
    settings = load_settings()
    prefix = notify_prefix(settings)
    trading = settings.get("trading", {})
    cfg = TradeConfig(
        capital=float(trading.get("capital", 10000)),
        point_value=float(trading.get("point_value", 10)),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(settings.get("backtest", {}).get("slip_ticks", 1)),
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"status": "RUNNING", "updated_at": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )

    logger.info(f"{prefix} load 1m {FULL_START}..{FULL_END}")
    df_raw = load_1m_range(FULL_START, FULL_END)
    if df_raw.empty:
        REPORT_PATH.write_text(
            json.dumps({"status": "ERROR", "error": "no 1m bars"}, indent=2),
            encoding="utf-8",
        )
        return 1

    df, day_sessions = prepare_advanced_features(df_raw)
    ids = ["tmf_daily_secure", "tmf_daily_trend"]
    tracks: dict[str, Any] = {}

    for sid in ids:
        params = _strat_params(settings, sid)
        hl = evaluate_advanced_splits(df, day_sessions, cfg, params)
        close = evaluate_close_splits(df, day_sessions, cfg, params)
        claim = CLAIMED[sid]
        tracks[sid] = {
            "enabled": True,
            "params": params,
            "engine_hl_touch": hl,
            "engine_close_touch": close,
            "claimed": claim,
            "claim_vs_hl": {
                "pnl_diff": round(hl["total_pnl"] - claim["net_pnl"], 2),
                "wr_diff": round(hl["total_win_rate"] - claim["win_rate"], 4),
            },
        }
        logger.info(
            f"{prefix} {sid} HL total={hl['total_pnl']} WR={hl['total_win_rate']} "
            f"| close total={close['total_pnl']} WR={close['total_win_rate']}"
        )

    # 進場日重疊（研究 HL 引擎、全樣本）
    d_secure = entry_dates(df, day_sessions, cfg, tracks["tmf_daily_secure"]["params"])
    d_trend = entry_dates(df, day_sessions, cfg, tracks["tmf_daily_trend"]["params"])
    overlap = d_secure & d_trend
    only_secure = d_secure - d_trend
    only_trend = d_trend - d_secure

    # max_lots=1 排他下：同日衝突時只能做一口 → 取 trend（較寬進場）或 secure 誰先觸發
    # 因進場條件 trend ⊇ secure（僅 vol 較鬆），衝突日 = overlap，實務上幾乎總是同一根或 trend 更早/同時
    exclusive_note = (
        "secure 進場日應為 trend 子集（同訊號、僅 vol_spike 更嚴）。"
        "雙軌同日衝突時 max_lots=1 無法加總兩邊淨利。"
    )

    # 簡易互斥組合：overlap 日只跑 trend；only_trend 跑 trend；only_secure 理論應空
    # 實際 PnL 需重跑互斥引擎；此處給結構診斷即可
    report = {
        "status": "DONE",
        "brand": "tidemark",
        "updated_at": datetime.now().isoformat(),
        "range": {"full": [FULL_START, FULL_END], "IS_end": IS_END, "OOS_start": OOS_START},
        "n_day_sessions": len(day_sessions),
        "pass_rule": "IS_net>0 and OOS_net>0 and IS_trades>=20 and OOS_trades>=10",
        "tracks": tracks,
        "overlap": {
            "secure_entry_days": len(d_secure),
            "trend_entry_days": len(d_trend),
            "overlap_days": len(overlap),
            "only_secure_days": len(only_secure),
            "only_trend_days": len(only_trend),
            "secure_subset_of_trend": len(only_secure) == 0,
            "overlap_ratio_of_secure": round(len(overlap) / len(d_secure), 4) if d_secure else None,
            "note": exclusive_note,
        },
        "architecture_flags": {
            "ensemble_orchestrator": False,
            "run_backtest_uses": "tmf_v1_only",
            "paper_replay_default": "tmf_v1",
            "both_enabled_in_yaml": True,
            "max_lots_shared_across_tracks": False,
            "fill_model_research": "intrabar_high_low_touch",
            "fill_model_strategy": "close_touch",
        },
        "verdict": None,
    }

    # 自動 verdict
    bits = []
    for sid in ids:
        hl = tracks[sid]["engine_hl_touch"]
        cl = tracks[sid]["engine_close_touch"]
        if hl["pass"]:
            bits.append(f"{sid}: HL pass IS/OOS")
        else:
            bits.append(f"{sid}: HL FAIL gate")
        if cl["pass"]:
            bits.append(f"{sid}: close pass IS/OOS")
        else:
            bits.append(f"{sid}: close FAIL gate (正式風控較嚴)")
    if report["overlap"]["secure_subset_of_trend"]:
        bits.append("secure ⊆ trend（非互補雙軌；量能過濾子集）")
    else:
        bits.append("secure 有 trend 沒有的進場日（異常，需查）")
    bits.append("無 ensemble 編排器：YAML 雙 enabled ≠ 並行一口帳本")
    report["verdict"] = bits

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"{prefix} wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
