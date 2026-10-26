#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
從最早 1m 回測鎖策略，記錄虧損日，上帝視角（事後路徑）歸類局式，
並回測多套應對訊號（實盤規則仍無偷看）。→ outputs/loss_regimes_report.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.engine import BacktestResult
from src.backtest.fast_signals import FastSignalConfig, run_rsi_tp_fast, _long_only_tp_loop
from src.backtest.metrics import daily_pnl_stats, pair_round_trips, summarize
from src.core.config import load_settings, notify_prefix
from src.core.types import Side
from src.data.bars import load_1m_range
from src.data.local_store import LocalStore


def _cfg(stop: float = 300, tp_unused: float = 0) -> FastSignalConfig:
    return FastSignalConfig(
        capital=10000,
        point_value=10,
        fee_per_side=20,
        tax_rate=0.00002,
        slip_ticks=1,
        sessions=("day",),
        stop_points=stop,
        max_daily_loss=0,
        flatten_eod=True,
        allow_short=False,
    )


def _score(result: BacktestResult, capital: float = 10000) -> dict[str, Any]:
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


def god_view_loss_days(bars: pd.DataFrame, trips: list[dict]) -> list[dict[str, Any]]:
    m = bars.copy()
    m["ts"] = pd.to_datetime(m["ts"])
    m = m.set_index("ts").sort_index()
    by: dict[str, float] = defaultdict(float)
    for tr in trips:
        by[str(pd.Timestamp(tr["exit_ts"]).date())] += float(tr["pnl"])
    loss_days = {d for d, p in by.items() if p <= 0}
    rows = []
    for tr in trips:
        d = str(pd.Timestamp(tr["exit_ts"]).date())
        if d not in loss_days:
            continue
        et = pd.Timestamp(tr["entry_ts"])
        xt = pd.Timestamp(tr["exit_ts"])
        day = m[m.index.date == et.date()].between_time("08:45", "13:45")
        if day.empty:
            continue
        after = day.loc[et:]
        path = after.iloc[1:] if len(after) > 1 else after
        entry = float(tr["entry"])
        mfe = float(path["high"].max() - entry) if len(path) else 0.0
        mae = float(entry - path["low"].min()) if len(path) else 0.0
        hit_tp40 = bool((path["high"] >= entry + 40).any()) if len(path) else False
        hit_tp80 = bool((path["high"] >= entry + 80).any()) if len(path) else False
        short_hit_tp40 = bool((path["low"] <= entry - 40).any()) if len(path) else False
        day_open = float(day["open"].iloc[0])
        day_close = float(day["close"].iloc[-1])
        day_ret = day_close - day_open
        first30 = day.iloc[:30]
        am_range = float(first30["high"].max() - first30["low"].min())
        mins = (et.hour * 60 + et.minute) - (8 * 60 + 45)
        recover = None
        if str(tr["reason_exit"]).startswith("stop"):
            after_exit = day.loc[xt:]
            recover = bool((after_exit["high"] >= entry + 40).any()) if len(after_exit) else False

        # regime label (god view taxonomy)
        if str(tr["reason_exit"]).startswith("stop") and mins <= 30 and not hit_tp40:
            regime = "open_waterfall"  # 開盤後急殺，多單直接停損
        elif str(tr["reason_exit"]).startswith("stop") and short_hit_tp40 and not hit_tp40:
            regime = "wrong_way_crash"  # 該做空／不該做多
        elif str(tr["reason_exit"]) == "flatten_eod" and mins >= 180:
            regime = "late_entry_grind"  # 尾盤勉強進場，拖到收盤
        elif str(tr["reason_exit"]) == "flatten_eod" and not hit_tp40:
            regime = "dead_cat_no_follow"  # 反彈無續航，拖到收盤
        elif str(tr["reason_exit"]).startswith("stop") and hit_tp40:
            regime = "stop_before_tp"  # 路徑上先碰到停損（或同日曾可 TP）
        else:
            regime = "other_loss"

        # god advice (hindsight only — for research narrative)
        if regime in ("open_waterfall", "wrong_way_crash"):
            advice = "skip_long_or_invert_short"
        elif regime == "late_entry_grind":
            advice = "ban_entry_after_noon"
        elif regime == "dead_cat_no_follow":
            advice = "require_momentum_confirm_or_time_stop"
        elif hit_tp40 and str(tr["reason_exit"]).startswith("stop"):
            advice = "tighter_trail_or_no_reentry"
        else:
            advice = "skip"

        rows.append(
            {
                "day": d,
                "day_pnl": round(by[d], 2),
                "trade_pnl": round(float(tr["pnl"]), 2),
                "reason_exit": tr["reason_exit"],
                "entry_ts": str(et),
                "exit_ts": str(xt),
                "entry_mins_from_open": mins,
                "entry": entry,
                "exit": float(tr["exit"]),
                "mfe": round(mfe, 1),
                "mae": round(mae, 1),
                "hit_tp40_after_entry": hit_tp40,
                "hit_tp80_after_entry": hit_tp80,
                "short_would_hit_tp40": short_hit_tp40,
                "day_ret_points": round(day_ret, 1),
                "am30_range": round(am_range, 1),
                "recover_tp40_after_stop": recover,
                "regime": regime,
                "god_advice": advice,
            }
        )
    return rows


def _rsi_entry(close: pd.Series, period: int, low_th: float) -> np.ndarray:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50.0).values
    r1 = np.roll(rsi, 1)
    r1[0] = np.nan
    long_entry = (r1 < low_th) & (rsi >= low_th)
    long_entry[: period + 2] = False
    return long_entry


def run_script_a_baseline(bars: pd.DataFrame) -> BacktestResult:
    """局式A核心：現役 RSI reclaim + TP40 stop300。"""
    return run_rsi_tp_fast(bars, 28, 20.0, _cfg(300), take_profit_points=40.0)


def run_script_b_no_late(bars: pd.DataFrame) -> BacktestResult:
    """局式B：避開晚進場 — 進場須在開盤後 150 分鐘內（約 11:15 前）。"""
    df = bars.reset_index(drop=True)
    close = df["close"].astype(float)
    long_entry = _rsi_entry(close, 28, 20.0)
    ts = pd.to_datetime(df["ts"])
    mins = (ts.dt.hour * 60 + ts.dt.minute) - (8 * 60 + 45)
    long_entry = long_entry & (mins.values <= 150)
    return _long_only_tp_loop(
        bars, _cfg(300), long_entry, take_profit_points=40.0, reason_entry="rsi_no_late"
    )


def run_script_c_confirm(bars: pd.DataFrame) -> BacktestResult:
    """局式C：避開假反彈 — RSI reclaim 後下一根收盤仍上漲才進。"""
    df = bars.reset_index(drop=True)
    close = df["close"].astype(float)
    raw = _rsi_entry(close, 28, 20.0)
    c = close.values
    # signal at i-1, confirm at i: close[i] > close[i-1]
    long_entry = np.zeros(len(df), dtype=bool)
    pending = False
    for i in range(len(df)):
        if pending:
            if c[i] > c[i - 1]:
                long_entry[i] = True
            pending = False
        elif bool(raw[i]):
            pending = True
    return _long_only_tp_loop(
        bars, _cfg(300), long_entry, take_profit_points=40.0, reason_entry="rsi_confirm"
    )


def run_script_d_skip_open(bars: pd.DataFrame) -> BacktestResult:
    """局式D：避開開盤急殺 — 開盤後 15 分鐘內不進。"""
    df = bars.reset_index(drop=True)
    close = df["close"].astype(float)
    long_entry = _rsi_entry(close, 28, 20.0)
    ts = pd.to_datetime(df["ts"])
    mins = (ts.dt.hour * 60 + ts.dt.minute) - (8 * 60 + 45)
    long_entry = long_entry & (mins.values >= 15)
    return _long_only_tp_loop(
        bars, _cfg(300), long_entry, take_profit_points=40.0, reason_entry="rsi_skip_open"
    )


def run_script_e_trend_filter(bars: pd.DataFrame) -> BacktestResult:
    """局式E：下跌日少做多 — close > EMA60 才允許 RSI 進場。"""
    df = bars.reset_index(drop=True)
    close = df["close"].astype(float)
    ema = close.ewm(span=60, adjust=False).mean().values
    long_entry = _rsi_entry(close, 28, 20.0) & (close.values > ema)
    return _long_only_tp_loop(
        bars, _cfg(300), long_entry, take_profit_points=40.0, reason_entry="rsi_ema60"
    )


def run_script_f_time_stop(bars: pd.DataFrame) -> BacktestResult:
    """
    局式F：無續航則砍 — 進場後 30 根仍未 +20 點則平倉（另保留 TP40/stop300/eod）。
    """
    # implement custom loop
    df = bars.reset_index(drop=True)
    close = df["close"].astype(float)
    long_entry = _rsi_entry(close, 28, 20.0)
    cfg = _cfg(300)
    close_a = close.values
    ts = df["ts"].values
    sess = df["session"].astype(str).values if "session" in df.columns else np.array(["day"] * len(df))
    in_sess = np.array([s == "day" for s in sess])
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])
    equity = cfg.capital
    position = Side.FLAT
    entry = 0.0
    bars_held = 0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    tp, stop = 40.0, 300.0

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    for i in range(len(df)):
        px = float(close_a[i])
        now = ts[i]
        if position == Side.LONG:
            bars_held += 1
            if (entry - px) >= stop:
                f = px - 1
                net = (f - entry) * 10 - cost(f)
                equity += net
                trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": "stop_loss"})
                position = Side.FLAT
            elif (px - entry) >= tp:
                f = px - 1
                net = (f - entry) * 10 - cost(f)
                equity += net
                trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": "take_profit"})
                position = Side.FLAT
            elif bars_held >= 30 and (px - entry) < 20:
                f = px - 1
                net = (f - entry) * 10 - cost(f)
                equity += net
                trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": "time_stop_30"})
                position = Side.FLAT
        if in_sess[i] and last_in_block[i] and position == Side.LONG:
            f = px - 1
            net = (f - entry) * 10 - cost(f)
            equity += net
            trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": "flatten_eod"})
            position = Side.FLAT
        if in_sess[i] and not last_in_block[i] and position == Side.FLAT and bool(long_entry[i]):
            f = px + 1
            equity -= cost(f)
            entry = f
            bars_held = 0
            position = Side.LONG
            trades.append({"ts": now, "action": "BUY", "price": f, "reason": "rsi_time_stop"})
        mtm = equity + ((px - entry) * 10 if position == Side.LONG else 0)
        if i % 10 == 0:
            curve.append({"ts": now, "equity": mtm, "price": px})
    return BacktestResult(
        equity_curve=curve or [{"ts": 0, "equity": equity, "price": 0}],
        trades=trades,
        final_equity=equity,
        n_steps=len(df),
    )


SCRIPTS = {
    "A_baseline_rsi": {
        "desc": "現役：RSI28 reclaim>20，TP40，stop300，日結",
        "fn": run_script_a_baseline,
        "regime_target": "default_mean_revert",
    },
    "B_no_late_entry": {
        "desc": "避開 late_entry_grind：11:15 前才進",
        "fn": run_script_b_no_late,
        "regime_target": "late_entry_grind",
    },
    "C_next_bar_confirm": {
        "desc": "避開 dead_cat：reclaim 後下一根續漲才進",
        "fn": run_script_c_confirm,
        "regime_target": "dead_cat_no_follow",
    },
    "D_skip_open_15m": {
        "desc": "避開 open_waterfall：開盤 15 分鐘內不進",
        "fn": run_script_d_skip_open,
        "regime_target": "open_waterfall",
    },
    "E_ema60_trend": {
        "desc": "避開下跌日：close>EMA60 才進",
        "fn": run_script_e_trend_filter,
        "regime_target": "wrong_way_crash",
    },
    "F_time_stop_30": {
        "desc": "無續航：30 根未 +20 則砍",
        "fn": run_script_f_time_stop,
        "regime_target": "dead_cat_no_follow",
    },
}


def main() -> int:
    settings = load_settings()
    store = LocalStore(settings)
    dates = store.list_dates("TMF", "1m")
    start, end = dates[0], dates[-1]
    is_end, oos_start = "2026-03-31", "2026-04-01"

    report_path = ROOT / "outputs" / "loss_regimes_report.json"
    report_path.write_text(
        json.dumps({"status": "RUNNING", "range": [start, end]}, indent=2), encoding="utf-8"
    )

    full = load_1m_range(start, end)
    is_bars = load_1m_range(start, is_end)
    oos_bars = load_1m_range(oos_start, end)

    base = run_script_a_baseline(full)
    trips = pair_round_trips(base.trades)
    loss_detail = god_view_loss_days(full, trips)
    regime_counts: dict[str, int] = defaultdict(int)
    for row in loss_detail:
        regime_counts[row["regime"]] += 1

    logger.info(
        f"{notify_prefix()} full {start}..{end} trips={len(trips)} "
        f"loss_rows={len(loss_detail)} regimes={dict(regime_counts)}"
    )

    script_results = []
    for name, meta in SCRIPTS.items():
        is_r = meta["fn"](is_bars)
        oos_r = meta["fn"](oos_bars)
        full_r = meta["fn"](full)
        is_s, oos_s, full_s = _score(is_r), _score(oos_r), _score(full_r)
        # which baseline loss days avoided on full?
        full_trips = pair_round_trips(full_r.trades)
        by = defaultdict(float)
        for t in full_trips:
            by[str(pd.Timestamp(t["exit_ts"]).date())] += float(t["pnl"])
        base_loss_days = {r["day"] for r in loss_detail}
        still_loss = sorted(d for d in base_loss_days if by.get(d, 0) <= 0 and d in by)
        avoided = sorted(d for d in base_loss_days if d not in by or by.get(d, 0) > 0)
        # avoided = was loss in baseline, now either no trade that day or win
        avoided_clean = []
        for d in base_loss_days:
            if d not in by:
                avoided_clean.append({"day": d, "how": "no_trade"})
            elif by[d] > 0:
                avoided_clean.append({"day": d, "how": "turned_profit", "pnl": round(by[d], 2)})
        script_results.append(
            {
                "id": name,
                "desc": meta["desc"],
                "regime_target": meta["regime_target"],
                "IS": is_s,
                "OOS": oos_s,
                "FULL": full_s,
                "baseline_loss_days_avoided": avoided_clean,
                "baseline_loss_days_still_loss_or_traded_red": [
                    d for d in base_loss_days if d in by and by[d] <= 0
                ],
                "pass_both_pos": is_s["net_pnl"] > 0 and oos_s["net_pnl"] > 0,
                "pass_dwr_0.8": bool(
                    is_s["net_pnl"] > 0
                    and oos_s["net_pnl"] > 0
                    and (is_s.get("daily_win_rate") or 0) > 0.8
                    and (oos_s.get("daily_win_rate") or 0) > 0.8
                ),
            }
        )
        logger.info(
            f"{notify_prefix()} {name} IS={is_s['net_pnl']} OOS={oos_s['net_pnl']} "
            f"dWR={is_s.get('daily_win_rate')}/{oos_s.get('daily_win_rate')} "
            f"avoided={len(avoided_clean)}"
        )

    script_results.sort(
        key=lambda r: (
            r["pass_dwr_0.8"],
            r["pass_both_pos"],
            r["OOS"].get("daily_win_rate") or 0,
            r["OOS"]["net_pnl"],
        ),
        reverse=True,
    )

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "bar_range": [start, end],
        "is_range": [start, is_end],
        "oos_range": [oos_start, end],
        "baseline_full": _score(base),
        "loss_day_god_view": loss_detail,
        "regime_counts": dict(regime_counts),
        "scripts": script_results,
        "note": (
            "上帝視角僅用於歸類虧損局式與設計腳本；"
            "各 script 回測規則只用 ≤t 資料。"
            "多套腳本對應不同局式，尚未做自動切換（regime router）。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # also drop portable signal recipe file for agents
    recipes = {
        "updated_at": payload["updated_at"],
        "regimes": {
            "open_waterfall": {
                "script": "D_skip_open_15m",
                "rule": "skip RSI long in first 15 minutes of day session",
            },
            "late_entry_grind": {
                "script": "B_no_late_entry",
                "rule": "only allow RSI long if minutes_from_open <= 150",
            },
            "dead_cat_no_follow": {
                "script": "C_next_bar_confirm or F_time_stop_30",
                "rule": "confirm next bar up, or time-stop if no +20 in 30 bars",
            },
            "wrong_way_crash": {
                "script": "E_ema60_trend",
                "rule": "require close > EMA60 for long mean-revert",
            },
            "default_mean_revert": {
                "script": "A_baseline_rsi",
                "rule": "RSI28 reclaim>20 TP40 stop300 flatten_eod",
            },
        },
        "report": str(report_path),
    }
    (ROOT / "outputs" / "regime_signal_recipes.json").write_text(
        json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"{notify_prefix()} wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
