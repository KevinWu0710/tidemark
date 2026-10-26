#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
上帝視角 → 規則 → 無偷看回測（日頻軌道研究）。

流程：
  1) 僅在 IS：用未來路徑標註「若在此進場，會先碰到 TP 還是停損」
     （標註可偷看；僅供挖規則）
  2) 在理想進場點上統計因果特徵（RSI／時段／夜盤／均線…）
  3) 抽出簡單規則（進場當下可得資訊）
  4) 規則以無偷看引擎在 IS/OOS 回測；日夜勝率分開

不過關不 enable tmf_daily_v1；不改 tmf_v1 鎖。
結果 → outputs/god_view_daily_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.fast_signals import (
    FastSignalConfig,
    _dual_tp_loop,
    _long_only_tp_loop,
    _rsi_series,
)
from src.backtest.metrics import (
    daily_pnl_stats,
    pair_round_trips,
    passes_session_daily_win_rate,
    summarize,
)
from src.core.config import load_settings, notify_prefix
from src.data.bars import load_1m_range
from src.data.sessions import label_session


DAY_OPEN_MIN = 8 * 60 + 45
TP = 40.0
STOP = 300.0


def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.reset_index(drop=True).copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if "session" not in df.columns:
        df["session"] = df["ts"].map(label_session)
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["mins"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute - DAY_OPEN_MIN
    close = df["close"].astype(float)
    for p in (6, 14, 28):
        df[f"rsi{p}"] = _rsi_series(close, p)
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema60"] = close.ewm(span=60, adjust=False).mean()
    # local bounce: close is min of last 5 (causal)
    df["roll_min5"] = close.rolling(5, min_periods=5).min()
    df["roll_max5"] = close.rolling(5, min_periods=5).max()
    df["is_local_low"] = close <= df["roll_min5"]
    df["is_local_high"] = close >= df["roll_max5"]
    # RSI reclaim features
    for p, th in ((6, 25.0), (14, 30.0), (28, 20.0)):
        r = df[f"rsi{p}"].values
        r1 = np.roll(r, 1)
        r1[0] = np.nan
        df[f"rsi{p}_reclaim_{int(th)}"] = (r1 < th) & (r >= th)
    for p, th in ((6, 75.0), (14, 70.0), (28, 80.0)):
        r = df[f"rsi{p}"].values
        r1 = np.roll(r, 1)
        r1[0] = np.nan
        df[f"rsi{p}_fail_{int(th)}"] = (r1 > th) & (r <= th)

    # night_ret / gap for day bars
    day = df[df["session"] == "day"]
    night = df[df["session"] == "night"]
    day_open = day.groupby("date")["open"].first()
    day_close = day.groupby("date")["close"].last()
    dates = sorted(day_open.index.tolist())
    night_ret: dict[str, float] = {}
    gap: dict[str, float] = {}
    for i, d in enumerate(dates):
        d_open_ts = day.loc[day["date"] == d, "ts"].iloc[0]
        if i == 0:
            nseg = night[night["ts"] < d_open_ts]
            gap[d] = 0.0
        else:
            prev = dates[i - 1]
            prev_last = day.loc[day["date"] == prev, "ts"].iloc[-1]
            nseg = night[(night["ts"] > prev_last) & (night["ts"] < d_open_ts)]
            gap[d] = float(day_open[d] - day_close[prev])
        if len(nseg) >= 2:
            night_ret[d] = float(nseg["close"].iloc[-1] - nseg["open"].iloc[0])
        else:
            night_ret[d] = 0.0
    df["night_ret"] = df["date"].map(night_ret).fillna(0.0)
    df["gap_pts"] = df["date"].map(gap).fillna(0.0)
    df["above_ema20"] = close > df["ema20"]
    df["above_ema60"] = close > df["ema60"]
    return df


def _session_blocks(df: pd.DataFrame) -> list[tuple[str, str, int, int]]:
    """List (date, session, start_i, end_i exclusive) contiguous session runs."""
    blocks: list[tuple[str, str, int, int]] = []
    n = len(df)
    i = 0
    sess = df["session"].astype(str).values
    dates = df["date"].values
    while i < n:
        if sess[i] not in ("day", "night"):
            i += 1
            continue
        s0, d0 = sess[i], dates[i]
        j = i + 1
        while j < n and sess[j] == s0 and dates[j] == d0:
            j += 1
        # night 15:00 may be date D but trading continues — keep calendar date of block start
        if j - i >= 10:
            blocks.append((str(d0), str(s0), i, j))
        i = j
    return blocks


def _god_label_candidates(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    cand: np.ndarray,
    start: int,
    end: int,
    *,
    tp: float,
    stop: float,
    side: str,
) -> np.ndarray:
    """只對候選 bar 做上帝視角（加速）；非候選維持 False。"""
    good = np.zeros(end - start, dtype=bool)
    for i in range(start, end - 1):
        if not bool(cand[i]):
            continue
        entry = float(close[i])
        hit = False
        for j in range(i + 1, end):
            if side == "long":
                if float(low[j]) <= entry - stop:
                    break
                if float(high[j]) >= entry + tp:
                    hit = True
                    break
            else:
                if float(high[j]) >= entry + stop:
                    break
                if float(low[j]) <= entry - tp:
                    hit = True
                    break
        good[i - start] = hit
    return good


def _label_is(df: pd.DataFrame, tp: float, stop: float) -> pd.DataFrame:
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    # 候選：局部轉折或 RSI 穿越（減少 O(L²)）
    cand_l = (
        df["is_local_low"].astype(bool).values
        | df["rsi28_reclaim_20"].astype(bool).values
        | df["rsi14_reclaim_30"].astype(bool).values
        | df["rsi6_reclaim_25"].astype(bool).values
    )
    cand_s = (
        df["is_local_high"].astype(bool).values
        | df["rsi28_fail_80"].astype(bool).values
        | df["rsi14_fail_70"].astype(bool).values
        | df["rsi6_fail_75"].astype(bool).values
    )
    good_long = np.zeros(len(df), dtype=bool)
    good_short = np.zeros(len(df), dtype=bool)
    blocks = _session_blocks(df)
    for bi, (_d, _s, a, b) in enumerate(blocks):
        gl = _god_label_candidates(
            high, low, close, cand_l, a, b, tp=tp, stop=stop, side="long"
        )
        gs = _god_label_candidates(
            high, low, close, cand_s, a, b, tp=tp, stop=stop, side="short"
        )
        good_long[a : a + len(gl)] = gl
        good_short[a : a + len(gs)] = gs
        if (bi + 1) % 80 == 0:
            logger.info(f"{notify_prefix()} god-label blocks {bi+1}/{len(blocks)}")
    out = df.copy()
    out["god_long"] = good_long
    out["god_short"] = good_short
    out["god_cand_long"] = cand_l
    out["god_cand_short"] = cand_s
    return out


def _feature_summary(df: pd.DataFrame, mask: np.ndarray, label: str) -> dict[str, Any]:
    sub = df.loc[mask]
    if sub.empty:
        return {"n": 0, "label": label}
    def q(col: str) -> dict[str, float]:
        s = sub[col].astype(float)
        return {
            "p25": round(float(s.quantile(0.25)), 2),
            "p50": round(float(s.quantile(0.50)), 2),
            "p75": round(float(s.quantile(0.75)), 2),
            "mean": round(float(s.mean()), 2),
        }

    sess_c = Counter(sub["session"].astype(str))
    return {
        "n": int(len(sub)),
        "label": label,
        "session_counts": dict(sess_c),
        "mins": q("mins") if "mins" in sub else {},
        "rsi28": q("rsi28"),
        "rsi14": q("rsi14"),
        "night_ret": q("night_ret"),
        "gap_pts": q("gap_pts"),
        "frac_above_ema20": round(float(sub["above_ema20"].mean()), 3),
        "frac_above_ema60": round(float(sub["above_ema60"].mean()), 3),
        "frac_local_low": round(float(sub["is_local_low"].mean()), 3),
        "frac_local_high": round(float(sub["is_local_high"].mean()), 3),
        "frac_rsi28_reclaim20": round(float(sub["rsi28_reclaim_20"].mean()), 3),
        "frac_rsi14_reclaim30": round(float(sub["rsi14_reclaim_30"].mean()), 3),
        "frac_rsi28_fail80": round(float(sub["rsi28_fail_80"].mean()), 3),
    }


def _rule_catalog() -> list[dict[str, Any]]:
    """因果規則：進場當下可得。"""
    rules: list[dict[str, Any]] = []

    def add(rid: str, side: str, desc: str, cols_all: list[str] | None = None, **kw: Any):
        rules.append({"id": rid, "side": side, "desc": desc, "cols_all": cols_all or [], **kw})

    # long recipes from typical god-view clusters
    add(
        "L_rsi28_reclaim20",
        "long",
        "RSI28 reclaim>20（現役鎖同源）",
        cols_all=["rsi28_reclaim_20"],
    )
    add(
        "L_rsi14_reclaim30",
        "long",
        "RSI14 reclaim>30",
        cols_all=["rsi14_reclaim_30"],
    )
    add(
        "L_rsi6_reclaim25",
        "long",
        "RSI6 reclaim>25（較密）",
        cols_all=["rsi6_reclaim_25"],
    )
    add(
        "L_rsi28_reclaim20_ema20",
        "long",
        "RSI28 reclaim20 + close>EMA20",
        cols_all=["rsi28_reclaim_20", "above_ema20"],
    )
    add(
        "L_rsi28_reclaim20_mins150",
        "long",
        "RSI28 reclaim20 + 開盤後≤150分",
        cols_all=["rsi28_reclaim_20"],
        max_mins=150,
    )
    add(
        "L_rsi28_reclaim20_night_ok",
        "long",
        "RSI28 reclaim20 + 夜盤>-120",
        cols_all=["rsi28_reclaim_20"],
        night_ret_gt=-120.0,
    )
    add(
        "L_local_low_rsi14_lt35",
        "long",
        "5棒低點 + RSI14<35",
        cols_all=["is_local_low"],
        rsi14_lt=35.0,
    )
    add(
        "L_local_low_rsi28_lt30_mins120",
        "long",
        "5棒低點 + RSI28<30 + ≤120分",
        cols_all=["is_local_low"],
        rsi28_lt=30.0,
        max_mins=120,
    )
    add(
        "L_rsi14_reclaim30_above_ema60",
        "long",
        "RSI14 reclaim30 + close>EMA60",
        cols_all=["rsi14_reclaim_30", "above_ema60"],
    )

    # short recipes
    add(
        "S_rsi28_fail80",
        "short",
        "RSI28 fail<80",
        cols_all=["rsi28_fail_80"],
    )
    add(
        "S_rsi14_fail70",
        "short",
        "RSI14 fail<70",
        cols_all=["rsi14_fail_70"],
    )
    add(
        "S_rsi6_fail75",
        "short",
        "RSI6 fail<75",
        cols_all=["rsi6_fail_75"],
    )
    add(
        "S_local_high_rsi14_gt65",
        "short",
        "5棒高點 + RSI14>65",
        cols_all=["is_local_high"],
        rsi14_gt=65.0,
    )
    add(
        "S_rsi28_fail80_below_ema20",
        "short",
        "RSI28 fail80 + close<EMA20",
        cols_all=["rsi28_fail_80"],
        below_ema20=True,
    )
    add(
        "S_rsi14_fail70_mins150",
        "short",
        "RSI14 fail70 + ≤150分",
        cols_all=["rsi14_fail_70"],
        max_mins=150,
    )
    return rules


def _eval_rule_mask(df: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    n = len(df)
    m = np.ones(n, dtype=bool)
    sess = df["session"].astype(str).values
    # only tradable sessions
    m &= np.isin(sess, ["day", "night"])
    for c in rule.get("cols_all") or []:
        m &= df[c].astype(bool).values
    if "max_mins" in rule:
        # day uses mins from 08:45; night: use minutes from block — approximate with clock
        # for night, mins is negative or large; only apply max_mins on day
        day = sess == "day"
        m &= (~day) | (df["mins"].values <= int(rule["max_mins"]))
    if "night_ret_gt" in rule:
        day = sess == "day"
        m &= (~day) | (df["night_ret"].values > float(rule["night_ret_gt"]))
    if "rsi14_lt" in rule:
        m &= df["rsi14"].values < float(rule["rsi14_lt"])
    if "rsi28_lt" in rule:
        m &= df["rsi28"].values < float(rule["rsi28_lt"])
    if "rsi14_gt" in rule:
        m &= df["rsi14"].values > float(rule["rsi14_gt"])
    if rule.get("below_ema20"):
        m &= ~df["above_ema20"].astype(bool).values
    return m


def _score_rule_on_labels(df: pd.DataFrame, mask: np.ndarray, side: str) -> dict[str, Any]:
    god = df["god_long"].values if side == "long" else df["god_short"].values
    fire = int(mask.sum())
    hit = int((mask & god).sum())
    god_n = int(god.sum())
    prec = hit / fire if fire else 0.0
    rec = hit / god_n if god_n else 0.0
    return {
        "n_fire": fire,
        "n_hit_god": hit,
        "n_god": god_n,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec > 0 else 0.0,
    }


def _cfg(trading: dict, slip: float, allow_short: bool) -> FastSignalConfig:
    return FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=("day", "night"),
        stop_points=STOP,
        flatten_eod=True,
        allow_short=allow_short,
    )


def _backtest_rule(df: pd.DataFrame, rule: dict[str, Any], trading: dict, slip: float):
    mask = _eval_rule_mask(df, rule)
    z = np.zeros(len(df), dtype=bool)
    if rule["side"] == "long":
        cfg = _cfg(trading, slip, allow_short=False)
        return _long_only_tp_loop(
            df, cfg, mask, take_profit_points=TP, reason_entry=rule["id"]
        )
    cfg = _cfg(trading, slip, allow_short=True)
    return _dual_tp_loop(
        df,
        cfg,
        z,
        mask,
        take_profit_long=0.0,
        take_profit_short=TP,
        stop_long=0.0,
        stop_short=STOP,
        reason_long="unused",
        reason_short=rule["id"],
        prefer="short",
    )


def _backtest_combo(
    df: pd.DataFrame,
    long_rule: dict[str, Any],
    short_rule: dict[str, Any],
    trading: dict,
    slip: float,
):
    lm = _eval_rule_mask(df, long_rule)
    sm = _eval_rule_mask(df, short_rule)
    cfg = _cfg(trading, slip, allow_short=True)
    return _dual_tp_loop(
        df,
        cfg,
        lm,
        sm,
        take_profit_long=TP,
        take_profit_short=TP,
        stop_long=STOP,
        stop_short=STOP,
        reason_long=long_rule["id"],
        reason_short=short_rule["id"],
        prefer="long",
    )


def _bt_score(result, capital: float, available: set[str]) -> dict[str, Any]:
    s = summarize(result, capital=capital, label="x")
    trips = pair_round_trips(result.trades)
    d = daily_pnl_stats(trips, session=None)
    n_avail = len(available)
    n_td = int(d["n_trade_days"] or 0)
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "max_drawdown": s["max_drawdown"],
        "day_coverage": round(n_td / n_avail, 4) if n_avail else 0.0,
        "avg_trips_per_trade_day": round(len(trips) / n_td, 3) if n_td else 0.0,
        "day_daily_win_rate": s.get("day_daily_win_rate"),
        "night_daily_win_rate": s.get("night_daily_win_rate"),
        "by_session": s.get("by_session"),
    }


def _available(bars: pd.DataFrame) -> set[str]:
    df = bars
    if "session" in df.columns:
        df = df[df["session"].astype(str).isin(["day", "night"])]
    return set(pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d").tolist())


def main() -> int:
    p = argparse.ArgumentParser(description="god-view daily entries → causal rules → BT")
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument("--tp", type=float, default=40.0)
    p.add_argument("--stop", type=float, default=300.0)
    p.add_argument("--min-daily-win-rate", type=float, default=0.55)
    p.add_argument("--min-coverage", type=float, default=0.3)
    args = p.parse_args()
    global TP, STOP
    TP, STOP = float(args.tp), float(args.stop)
    min_dwr = float(args.min_daily_win_rate)
    min_cov = float(args.min_coverage)

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    capital = float(trading["capital"])

    report_path = ROOT / "outputs" / "god_view_daily_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"status": "RUNNING", "updated_at": datetime.now().isoformat(timespec="seconds")}, indent=2),
        encoding="utf-8",
    )

    logger.info(f"{notify_prefix()} god-view daily: prepare IS labels tp={TP} stop={STOP}")
    is_df = _label_is(_prepare(load_1m_range(args.is_start, args.is_end)), TP, STOP)
    oos_df = _prepare(load_1m_range(args.oos_start, args.oos_end))
    is_avail = _available(is_df)
    oos_avail = _available(oos_df)

    n_long = int(is_df["god_long"].sum())
    n_short = int(is_df["god_short"].sum())
    # per-day: any god opportunity
    by_day_long = is_df.loc[is_df["god_long"]].groupby("date").size()
    by_day_short = is_df.loc[is_df["god_short"]].groupby("date").size()
    god_stats = {
        "n_god_long_bars": n_long,
        "n_god_short_bars": n_short,
        "frac_bars_god_long": round(n_long / max(len(is_df), 1), 4),
        "frac_bars_god_short": round(n_short / max(len(is_df), 1), 4),
        "days_with_god_long": int((by_day_long > 0).sum()),
        "days_with_god_short": int((by_day_short > 0).sum()),
        "n_is_days": len(is_avail),
        "god_long_feature_summary": _feature_summary(
            is_df, is_df["god_long"].values, "god_long"
        ),
        "god_short_feature_summary": _feature_summary(
            is_df, is_df["god_short"].values, "god_short"
        ),
    }
    logger.info(
        f"{notify_prefix()} god labels long_bars={n_long} short_bars={n_short} "
        f"days_L={god_stats['days_with_god_long']} days_S={god_stats['days_with_god_short']}"
    )

    rules = _rule_catalog()
    mined: list[dict[str, Any]] = []
    for rule in rules:
        mask = _eval_rule_mask(is_df, rule)
        lab = _score_rule_on_labels(is_df, mask, rule["side"])
        mined.append({**rule, "is_label_fit": lab})
    mined.sort(key=lambda r: (r["is_label_fit"]["f1"], r["is_label_fit"]["precision"]), reverse=True)
    logger.info(f"{notify_prefix()} mined {len(mined)} rules; backtesting…")

    bt_rows: list[dict[str, Any]] = []
    for i, rule in enumerate(mined):
        is_s = _bt_score(_backtest_rule(is_df, rule, trading, slip), capital, is_avail)
        oos_s = _bt_score(_backtest_rule(oos_df, rule, trading, slip), capital, oos_avail)
        edge = bool(
            is_s["net_pnl"] > 0
            and oos_s["net_pnl"] > 0
            and passes_session_daily_win_rate(
                is_s.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
            )
            and passes_session_daily_win_rate(
                oos_s.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
            )
            and is_s["day_coverage"] >= min_cov
            and oos_s["day_coverage"] >= min_cov
            and is_s["n_round_trips"] >= 20
            and oos_s["n_round_trips"] >= 10
        )
        bt_rows.append(
            {
                "id": rule["id"],
                "side": rule["side"],
                "desc": rule["desc"],
                "is_label_fit": rule["is_label_fit"],
                "IS": is_s,
                "OOS": oos_s,
                "edge_ok": edge,
                "pass": edge,
            }
        )
        if (i + 1) % 5 == 0 or i + 1 == len(mined):
            logger.info(f"{notify_prefix()} god-rule BT {i+1}/{len(mined)}")

    # dual: top long × top short by OOS pnl among positive IS
    longs = [r for r in bt_rows if r["side"] == "long"]
    shorts = [r for r in bt_rows if r["side"] == "short"]
    longs_s = sorted(longs, key=lambda r: (r["IS"]["net_pnl"] > 0, r["OOS"]["net_pnl"]), reverse=True)[:4]
    shorts_s = sorted(shorts, key=lambda r: (r["IS"]["net_pnl"] > 0, r["OOS"]["net_pnl"]), reverse=True)[:4]
    dual_rows: list[dict[str, Any]] = []
    rule_by_id = {r["id"]: r for r in rules}
    for L in longs_s:
        for S in shorts_s:
            lr, sr = rule_by_id[L["id"]], rule_by_id[S["id"]]
            is_s = _bt_score(
                _backtest_combo(is_df, lr, sr, trading, slip), capital, is_avail
            )
            oos_s = _bt_score(
                _backtest_combo(oos_df, lr, sr, trading, slip), capital, oos_avail
            )
            edge = bool(
                is_s["net_pnl"] > 0
                and oos_s["net_pnl"] > 0
                and passes_session_daily_win_rate(
                    is_s.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
                )
                and passes_session_daily_win_rate(
                    oos_s.get("by_session") or {}, min_dwr, require_sessions=["day", "night"]
                )
                and is_s["day_coverage"] >= min_cov
                and oos_s["day_coverage"] >= min_cov
            )
            dual_rows.append(
                {
                    "id": f"{L['id']}__{S['id']}",
                    "side": "dual",
                    "long": L["id"],
                    "short": S["id"],
                    "IS": is_s,
                    "OOS": oos_s,
                    "edge_ok": edge,
                    "pass": edge,
                }
            )
    dual_rows.sort(
        key=lambda r: (r["edge_ok"], r["OOS"]["day_coverage"], r["OOS"]["net_pnl"]),
        reverse=True,
    )
    bt_rows.sort(
        key=lambda r: (r["edge_ok"], r["OOS"]["day_coverage"], r["OOS"]["net_pnl"]),
        reverse=True,
    )
    winners = [r for r in bt_rows + dual_rows if r["pass"]]

    # narrative recipes from god view
    gl = god_stats["god_long_feature_summary"]
    gs = god_stats["god_short_feature_summary"]
    recipes = {
        "long": [
            f"理想多點 RSI28 中位≈{gl.get('rsi28', {}).get('p50')}；"
            f"reclaim20 占比 {gl.get('frac_rsi28_reclaim20')}",
            f"時段 mins 中位≈{gl.get('mins', {}).get('p50')}；"
            f"夜盤報酬中位≈{gl.get('night_ret', {}).get('p50')}",
            f"站上 EMA20 占比 {gl.get('frac_above_ema20')}；local_low 占比 {gl.get('frac_local_low')}",
        ],
        "short": [
            f"理想空點 RSI28 中位≈{gs.get('rsi28', {}).get('p50')}；"
            f"fail80 占比 {gs.get('frac_rsi28_fail80')}",
            f"local_high 占比 {gs.get('frac_local_high')}；"
            f"站上 EMA20 占比 {gs.get('frac_above_ema20')}",
        ],
    }

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "god_view_label_IS_then_causal_rule_BT",
        "tp_points": TP,
        "stop_points": STOP,
        "min_daily_win_rate": min_dwr,
        "min_coverage": min_cov,
        "god_view_IS": god_stats,
        "recipes_from_god_view": recipes,
        "rules_ranked_by_label_f1": [
            {"id": r["id"], "side": r["side"], "desc": r["desc"], **r["is_label_fit"]}
            for r in mined[:12]
        ],
        "n_pass": len(winners),
        "winners": winners[:10],
        "top10_bt": bt_rows[:10],
        "top5_dual": dual_rows[:5],
        "note": (
            "上帝視角只用於 IS 標註與敘事；回測規則無偷看。"
            "過關才考慮填 tmf_daily_v1；否則維持 tmf_v1 鎖。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} god-view daily done pass={len(winners)} "
        f"best_bt={bt_rows[0]['id'] if bt_rows else None} -> {report_path}"
    )
    if winners:
        w = winners[0]
        logger.info(
            f"  WINNER {w['id']} cov={w['OOS']['day_coverage']} "
            f"IS={w['IS']['net_pnl']} OOS={w['OOS']['net_pnl']}"
        )
    elif bt_rows:
        b = bt_rows[0]
        logger.info(
            f"  BEST_NONPASS {b['id']} IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']} "
            f"cov={b['OOS']['day_coverage']}"
        )
    return 0 if winners else 2


if __name__ == "__main__":
    raise SystemExit(main())
