#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
從上帝視角「每日理想多空點」挖日頻規則／訊號（不含現役 RSI28 reclaim）。

依據既有 god-view 特徵：
  - 理想多 ≈ 5棒低點（~89%）＋ RSI14 中位帶 ~35–49 ＋ 多數在 EMA20 下
  - 理想空 ≈ 5棒高點（~89%）＋ RSI14 中位帶 ~52–65 ＋ 多數在 EMA20 上

流程：
  1) IS：路徑統計（MAE／到 TP 棒數）→ 建議 TP／stop 網格
  2) 規則目錄：局部高低＋RSI 帶＋EMA（無 RSI reclaim／fail 穿越族）
  3) 規則 × TP × stop 無偷看 IS/OOS；雙邊組合

結果 → outputs/daily_from_god_report.json
不過關不 enable tmf_daily_v1。
"""
from __future__ import annotations

import argparse
import json
import sys
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
REPORT = ROOT / "outputs" / "daily_from_god_report.json"


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
    df["above_ema20"] = close > df["ema20"]
    df["above_ema60"] = close > df["ema60"]
    for w in (3, 5, 8):
        df[f"roll_min{w}"] = close.rolling(w, min_periods=w).min()
        df[f"roll_max{w}"] = close.rolling(w, min_periods=w).max()
        df[f"local_low{w}"] = close <= df[f"roll_min{w}"]
        df[f"local_high{w}"] = close >= df[f"roll_max{w}"]
    # bounce confirm：前一根是 5棒低，本根收高
    prev_ll = df["local_low5"].shift(1).fillna(False).astype(bool)
    df["bounce_up"] = prev_ll & (close > close.shift(1))
    prev_hh = df["local_high5"].shift(1).fillna(False).astype(bool)
    df["fade_down"] = prev_hh & (close < close.shift(1))

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
    return df


def _session_blocks(df: pd.DataFrame) -> list[tuple[str, str, int, int]]:
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
        if j - i >= 10:
            blocks.append((str(d0), str(s0), i, j))
        i = j
    return blocks


def _path_profile(
    df: pd.DataFrame,
    entry_mask: np.ndarray,
    side: str,
    *,
    tp: float,
    stop: float,
    max_sample: int = 8000,
) -> dict[str, Any]:
    """對進場候選做路徑統計（可偷看；只用於選 TP/stop）。"""
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    idxs = np.flatnonzero(entry_mask)
    if len(idxs) > max_sample:
        rng = np.random.default_rng(42)
        idxs = np.sort(rng.choice(idxs, size=max_sample, replace=False))
    # map bar → block end
    end_of = np.full(len(df), len(df), dtype=np.int64)
    for _d, _s, a, b in _session_blocks(df):
        end_of[a:b] = b

    hits = 0
    stops = 0
    timeouts = 0
    mae_hit: list[float] = []
    bars_hit: list[int] = []
    mfe_all: list[float] = []
    mae_all: list[float] = []

    for i in idxs:
        e = int(i)
        entry = float(close[e])
        end = int(end_of[e])
        mae = 0.0
        mfe = 0.0
        outcome = "timeout"
        bars = 0
        for j in range(e + 1, end):
            hj, lj = float(high[j]), float(low[j])
            if side == "long":
                mae = max(mae, entry - lj)
                mfe = max(mfe, hj - entry)
                if lj <= entry - stop:
                    outcome = "stop"
                    bars = j - e
                    break
                if hj >= entry + tp:
                    outcome = "tp"
                    bars = j - e
                    break
            else:
                mae = max(mae, hj - entry)
                mfe = max(mfe, entry - lj)
                if hj >= entry + stop:
                    outcome = "stop"
                    bars = j - e
                    break
                if lj <= entry - tp:
                    outcome = "tp"
                    bars = j - e
                    break
        mfe_all.append(mfe)
        mae_all.append(mae)
        if outcome == "tp":
            hits += 1
            mae_hit.append(mae)
            bars_hit.append(bars)
        elif outcome == "stop":
            stops += 1
        else:
            timeouts += 1

    n = max(len(idxs), 1)

    def _q(xs: list[float] | list[int]) -> dict[str, float]:
        if not xs:
            return {}
        a = np.asarray(xs, dtype=float)
        return {
            "p25": round(float(np.quantile(a, 0.25)), 2),
            "p50": round(float(np.quantile(a, 0.50)), 2),
            "p75": round(float(np.quantile(a, 0.75)), 2),
            "mean": round(float(a.mean()), 2),
        }

    return {
        "n_sample": int(len(idxs)),
        "hit_rate": round(hits / n, 4),
        "stop_rate": round(stops / n, 4),
        "timeout_rate": round(timeouts / n, 4),
        "mae_when_hit": _q(mae_hit),
        "bars_to_tp": _q(bars_hit),
        "mfe_all": _q(mfe_all),
        "mae_all": _q(mae_all),
        "tp": tp,
        "stop": stop,
        "side": side,
    }


def _rule_catalog() -> list[dict[str, Any]]:
    """日頻規則：局部高低＋RSI 中位帶＋EMA。不含 RSI reclaim／fail。"""
    rules: list[dict[str, Any]] = []

    def add(rid: str, side: str, desc: str, **kw: Any) -> None:
        rules.append({"id": rid, "side": side, "desc": desc, **kw})

    # —— long：貼近 god 多點特徵 ——
    add("L_ll5", "long", "5棒低點", col="local_low5")
    add("L_ll3", "long", "3棒低點（較密）", col="local_low3")
    add("L_ll8", "long", "8棒低點（較稀）", col="local_low8")
    add(
        "L_ll5_rsi14_band",
        "long",
        "5棒低 + RSI14∈[35,50]（god 中位帶）",
        col="local_low5",
        rsi14_lo=35.0,
        rsi14_hi=50.0,
    )
    add(
        "L_ll5_rsi14_lt45",
        "long",
        "5棒低 + RSI14<45",
        col="local_low5",
        rsi14_hi=45.0,
    )
    add(
        "L_ll5_below_ema20",
        "long",
        "5棒低 + close<EMA20",
        col="local_low5",
        below_ema20=True,
    )
    add(
        "L_ll5_band_below_ema20",
        "long",
        "5棒低 + RSI14∈[35,50] + <EMA20",
        col="local_low5",
        rsi14_lo=35.0,
        rsi14_hi=50.0,
        below_ema20=True,
    )
    add(
        "L_bounce_up",
        "long",
        "前棒5低＋本棒收高（確認反彈）",
        col="bounce_up",
    )
    add(
        "L_bounce_rsi14_band",
        "long",
        "反彈確認 + RSI14∈[35,50]",
        col="bounce_up",
        rsi14_lo=35.0,
        rsi14_hi=50.0,
    )
    add(
        "L_ll5_rsi14_band_day_early",
        "long",
        "5棒低 + RSI帶 + 日盤≤150分（夜盤不限）",
        col="local_low5",
        rsi14_lo=35.0,
        rsi14_hi=50.0,
        max_mins_day=150,
    )
    add(
        "L_ll5_night_ok",
        "long",
        "5棒低 + 日盤夜報酬>-80",
        col="local_low5",
        night_ret_gt=-80.0,
    )

    # —— short：貼近 god 空點特徵 ——
    add("S_lh5", "short", "5棒高點", col="local_high5")
    add("S_lh3", "short", "3棒高點", col="local_high3")
    add("S_lh8", "short", "8棒高點", col="local_high8")
    add(
        "S_lh5_rsi14_band",
        "short",
        "5棒高 + RSI14∈[52,68]（god 中位帶）",
        col="local_high5",
        rsi14_lo=52.0,
        rsi14_hi=68.0,
    )
    add(
        "S_lh5_rsi14_gt55",
        "short",
        "5棒高 + RSI14>55",
        col="local_high5",
        rsi14_lo=55.0,
    )
    add(
        "S_lh5_above_ema20",
        "short",
        "5棒高 + close>EMA20",
        col="local_high5",
        above_ema20=True,
    )
    add(
        "S_lh5_band_above_ema20",
        "short",
        "5棒高 + RSI14∈[52,68] + >EMA20",
        col="local_high5",
        rsi14_lo=52.0,
        rsi14_hi=68.0,
        above_ema20=True,
    )
    add(
        "S_fade_down",
        "short",
        "前棒5高＋本棒收低（確認回落）",
        col="fade_down",
    )
    add(
        "S_fade_rsi14_band",
        "short",
        "回落確認 + RSI14∈[52,68]",
        col="fade_down",
        rsi14_lo=52.0,
        rsi14_hi=68.0,
    )
    return rules


def _eval_rule_mask(df: pd.DataFrame, rule: dict[str, Any]) -> np.ndarray:
    n = len(df)
    m = np.ones(n, dtype=bool)
    sess = df["session"].astype(str).values
    m &= np.isin(sess, ["day", "night"])
    col = rule.get("col")
    if col:
        m &= df[col].astype(bool).values
    r14 = df["rsi14"].astype(float).values
    if "rsi14_lo" in rule:
        m &= r14 >= float(rule["rsi14_lo"])
    if "rsi14_hi" in rule:
        m &= r14 <= float(rule["rsi14_hi"])
    if rule.get("below_ema20"):
        m &= ~df["above_ema20"].astype(bool).values
    if rule.get("above_ema20"):
        m &= df["above_ema20"].astype(bool).values
    if "max_mins_day" in rule:
        day = sess == "day"
        m &= (~day) | (df["mins"].values <= int(rule["max_mins_day"]))
    if "night_ret_gt" in rule:
        day = sess == "day"
        m &= (~day) | (df["night_ret"].values > float(rule["night_ret_gt"]))
    return m


def _cfg(trading: dict, slip: float, allow_short: bool, stop: float) -> FastSignalConfig:
    return FastSignalConfig(
        capital=float(trading["capital"]),
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=float(slip),
        sessions=("day", "night"),
        stop_points=float(stop),
        flatten_eod=True,
        allow_short=allow_short,
    )


def _bt_long(df: pd.DataFrame, mask: np.ndarray, trading: dict, slip: float, tp: float, stop: float, rid: str):
    cfg = _cfg(trading, slip, False, stop)
    return _long_only_tp_loop(df, cfg, mask, take_profit_points=tp, reason_entry=rid)


def _bt_short(df: pd.DataFrame, mask: np.ndarray, trading: dict, slip: float, tp: float, stop: float, rid: str):
    z = np.zeros(len(df), dtype=bool)
    cfg = _cfg(trading, slip, True, stop)
    return _dual_tp_loop(
        df,
        cfg,
        z,
        mask,
        take_profit_long=0.0,
        take_profit_short=tp,
        stop_long=0.0,
        stop_short=stop,
        reason_long="unused",
        reason_short=rid,
        prefer="short",
    )


def _bt_dual(
    df: pd.DataFrame,
    lm: np.ndarray,
    sm: np.ndarray,
    trading: dict,
    slip: float,
    tp: float,
    stop: float,
    lid: str,
    sid: str,
):
    cfg = _cfg(trading, slip, True, stop)
    return _dual_tp_loop(
        df,
        cfg,
        lm,
        sm,
        take_profit_long=tp,
        take_profit_short=tp,
        stop_long=stop,
        stop_short=stop,
        reason_long=lid,
        reason_short=sid,
        prefer="long",
    )


def _bt_score(result, capital: float, available: set[str]) -> dict[str, Any]:
    s = summarize(result, capital=capital, label="x")
    trips = pair_round_trips(result.trades)
    d = daily_pnl_stats(trips, session=None)
    n_avail = len(available)
    n_td = int(d["n_trade_days"] or 0)
    wr = None
    if trips:
        wins = sum(1 for t in trips if float(t.get("pnl") or 0) > 0)
        wr = round(wins / len(trips), 4)
    return {
        "net_pnl": s["net_pnl"],
        "n_round_trips": s["n_round_trips"],
        "max_drawdown": s["max_drawdown"],
        "trade_win_rate": wr,
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


def _pass_edge(
    is_s: dict[str, Any],
    oos_s: dict[str, Any],
    *,
    min_dwr: float,
    min_cov: float,
) -> bool:
    return bool(
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
        and is_s["n_round_trips"] >= 30
        and oos_s["n_round_trips"] >= 15
    )


def main() -> int:
    p = argparse.ArgumentParser(description="daily rules from god-view features (no RSI reclaim)")
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument("--min-daily-win-rate", type=float, default=0.52)
    p.add_argument("--min-coverage", type=float, default=0.5)
    args = p.parse_args()
    min_dwr = float(args.min_daily_win_rate)
    min_cov = float(args.min_coverage)

    settings = load_settings()
    trading = settings["trading"]
    slip = float(settings.get("backtest", {}).get("slip_ticks", 1))
    capital = float(trading["capital"])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {"status": "RUNNING", "updated_at": datetime.now().isoformat(timespec="seconds")},
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(f"{notify_prefix()} daily-from-god: prepare IS/OOS")
    is_df = _prepare(load_1m_range(args.is_start, args.is_end))
    oos_df = _prepare(load_1m_range(args.oos_start, args.oos_end))
    is_avail = _available(is_df)
    oos_avail = _available(oos_df)

    # —— 路徑剖面：局部高低在多組 TP/stop 下 ——
    probe_exits = [
        (15.0, 40.0),
        (20.0, 50.0),
        (20.0, 80.0),
        (25.0, 60.0),
        (25.0, 100.0),
        (30.0, 80.0),
        (30.0, 120.0),
        (40.0, 100.0),
        (40.0, 150.0),
        (40.0, 300.0),
    ]
    sess_ok = is_df["session"].astype(str).isin(["day", "night"]).values
    ll5 = is_df["local_low5"].astype(bool).values & sess_ok
    lh5 = is_df["local_high5"].astype(bool).values & sess_ok
    path_rows: list[dict[str, Any]] = []
    for tp, stop in probe_exits:
        pl = _path_profile(is_df, ll5, "long", tp=tp, stop=stop)
        ps = _path_profile(is_df, lh5, "short", tp=tp, stop=stop)
        path_rows.append({"entry": "local_low5", **pl})
        path_rows.append({"entry": "local_high5", **ps})
        logger.info(
            f"{notify_prefix()} path L_ll5 tp={tp} stop={stop} hit={pl['hit_rate']} "
            f"mae_hit_p50={pl['mae_when_hit'].get('p50')} bars_p50={pl['bars_to_tp'].get('p50')}"
        )

    # 建議 exit：hit_rate 高且 mae_when_hit.p75 < stop*0.6
    def _rank_path(r: dict[str, Any]) -> float:
        mae_p75 = float((r.get("mae_when_hit") or {}).get("p75") or 999)
        return float(r["hit_rate"]) - 0.001 * mae_p75

    long_paths = [r for r in path_rows if r["side"] == "long"]
    short_paths = [r for r in path_rows if r["side"] == "short"]
    long_paths.sort(key=_rank_path, reverse=True)
    short_paths.sort(key=_rank_path, reverse=True)
    suggested = {
        "long_top3": long_paths[:3],
        "short_top3": short_paths[:3],
        "note": "路徑統計可偷看；僅用來縮小 TP/stop 網格",
    }

    # BT 網格：路徑建議 + 日頻縮緊組
    exit_grid = sorted(
        {
            (15.0, 40.0),
            (20.0, 50.0),
            (20.0, 80.0),
            (25.0, 60.0),
            (25.0, 100.0),
            (30.0, 80.0),
            (30.0, 120.0),
            (40.0, 100.0),
            (40.0, 150.0),
            *((float(r["tp"]), float(r["stop"])) for r in long_paths[:3]),
            *((float(r["tp"]), float(r["stop"])) for r in short_paths[:3]),
        }
    )

    rules = _rule_catalog()
    logger.info(
        f"{notify_prefix()} rules={len(rules)} exits={len(exit_grid)} "
        f"combos≈{len(rules)*len(exit_grid)}"
    )

    bt_rows: list[dict[str, Any]] = []
    total = len(rules) * len(exit_grid)
    done = 0
    for rule in rules:
        mask_is = _eval_rule_mask(is_df, rule)
        mask_oos = _eval_rule_mask(oos_df, rule)
        for tp, stop in exit_grid:
            if rule["side"] == "long":
                is_r = _bt_long(is_df, mask_is, trading, slip, tp, stop, rule["id"])
                oos_r = _bt_long(oos_df, mask_oos, trading, slip, tp, stop, rule["id"])
            else:
                is_r = _bt_short(is_df, mask_is, trading, slip, tp, stop, rule["id"])
                oos_r = _bt_short(oos_df, mask_oos, trading, slip, tp, stop, rule["id"])
            is_s = _bt_score(is_r, capital, is_avail)
            oos_s = _bt_score(oos_r, capital, oos_avail)
            ok = _pass_edge(is_s, oos_s, min_dwr=min_dwr, min_cov=min_cov)
            bt_rows.append(
                {
                    "id": f"{rule['id']}_tp{int(tp)}_st{int(stop)}",
                    "rule": rule["id"],
                    "desc": rule["desc"],
                    "side": rule["side"],
                    "tp": tp,
                    "stop": stop,
                    "IS": is_s,
                    "OOS": oos_s,
                    "pass": ok,
                }
            )
            done += 1
            if done % 40 == 0 or done == total:
                logger.info(f"{notify_prefix()} daily-from-god BT {done}/{total}")

    bt_rows.sort(
        key=lambda r: (
            r["pass"],
            r["OOS"]["day_coverage"],
            r["OOS"]["net_pnl"],
            r["IS"]["net_pnl"],
        ),
        reverse=True,
    )

    # dual：IS>0 且 coverage 高的多／空各取 top
    longs = [
        r
        for r in bt_rows
        if r["side"] == "long" and r["IS"]["net_pnl"] > 0 and r["OOS"]["net_pnl"] > 0
    ]
    shorts = [
        r
        for r in bt_rows
        if r["side"] == "short" and r["IS"]["net_pnl"] > 0 and r["OOS"]["net_pnl"] > 0
    ]
    longs_s = sorted(longs, key=lambda r: (r["OOS"]["day_coverage"], r["OOS"]["net_pnl"]), reverse=True)[:5]
    shorts_s = sorted(shorts, key=lambda r: (r["OOS"]["day_coverage"], r["OOS"]["net_pnl"]), reverse=True)[:5]
    rule_by_id = {r["id"]: r for r in rules}
    dual_rows: list[dict[str, Any]] = []
    for L in longs_s:
        for S in shorts_s:
            # 同一 tp/stop 才合（避免混參數）
            if L["tp"] != S["tp"] or L["stop"] != S["stop"]:
                continue
            lr, sr = rule_by_id[L["rule"]], rule_by_id[S["rule"]]
            tp, stop = float(L["tp"]), float(L["stop"])
            is_r = _bt_dual(
                is_df,
                _eval_rule_mask(is_df, lr),
                _eval_rule_mask(is_df, sr),
                trading,
                slip,
                tp,
                stop,
                lr["id"],
                sr["id"],
            )
            oos_r = _bt_dual(
                oos_df,
                _eval_rule_mask(oos_df, lr),
                _eval_rule_mask(oos_df, sr),
                trading,
                slip,
                tp,
                stop,
                lr["id"],
                sr["id"],
            )
            is_s = _bt_score(is_r, capital, is_avail)
            oos_s = _bt_score(oos_r, capital, oos_avail)
            ok = _pass_edge(is_s, oos_s, min_dwr=min_dwr, min_cov=min_cov)
            dual_rows.append(
                {
                    "id": f"{L['rule']}__{S['rule']}_tp{int(tp)}_st{int(stop)}",
                    "side": "dual",
                    "long": L["rule"],
                    "short": S["rule"],
                    "tp": tp,
                    "stop": stop,
                    "IS": is_s,
                    "OOS": oos_s,
                    "pass": ok,
                }
            )
    dual_rows.sort(
        key=lambda r: (r["pass"], r["OOS"]["day_coverage"], r["OOS"]["net_pnl"]),
        reverse=True,
    )

    winners = [r for r in bt_rows + dual_rows if r["pass"]]
    # 敘事：從路徑＋規則抽出「每日訊號」候選（即使未過閘也列最佳日頻）
    daily_ranked = sorted(
        bt_rows,
        key=lambda r: (
            r["OOS"]["day_coverage"] >= 0.5,
            r["IS"]["net_pnl"] > 0 and r["OOS"]["net_pnl"] > 0,
            r["OOS"]["net_pnl"],
            r["OOS"]["day_coverage"],
        ),
        reverse=True,
    )

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "god_feature_clusters_path_exit_grid_causal_BT",
        "exclude": ["rsi28_reclaim", "rsi_fail_cross", "lock_compare"],
        "min_daily_win_rate": min_dwr,
        "min_coverage": min_cov,
        "god_feature_recipe": {
            "long": "local_low(~89%) + RSI14≈35–50 + 多在 EMA20 下；夜盤機會多",
            "short": "local_high(~89%) + RSI14≈52–68 + 多在 EMA20 上",
        },
        "path_suggested_exits": suggested,
        "n_pass": len(winners),
        "winners": winners[:15],
        "top15_single": bt_rows[:15],
        "top10_dual": dual_rows[:10],
        "best_daily_track": daily_ranked[:10],
        "note": (
            "規則來自 god 特徵簇（局部高低／RSI 中位帶／EMA），不含現役 RSI reclaim。"
            "過關才考慮 tmf_daily_v1；否則維持日頻研究軌道。"
        ),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} daily-from-god done pass={len(winners)} "
        f"best={bt_rows[0]['id'] if bt_rows else None} -> {REPORT}"
    )
    if winners:
        w = winners[0]
        logger.info(
            f"  WINNER {w['id']} cov={w['OOS']['day_coverage']} "
            f"IS={w['IS']['net_pnl']} OOS={w['OOS']['net_pnl']}"
        )
        return 0
    if daily_ranked:
        b = daily_ranked[0]
        logger.info(
            f"  best_daily_track {b['id']} cov={b['OOS']['day_coverage']} "
            f"IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']} (no gate pass)"
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
