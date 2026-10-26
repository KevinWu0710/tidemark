#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日頻稀訊號：從 god 特徵簇挖規則，但不強求每日多筆。

相對 dense 版：
  - 進場更挑（bounce／8棒／雙條件）
  - 冷卻棒數 + 每盤最多 1～2 筆（節流）
  - 覆蓋閘門放寬（預設 0.2）
  - 出場偏較好 R:R（停利不遠小於停損太多時靠篩選撐勝率）

不含 RSI28 reclaim。結果 → outputs/daily_from_god_sparse_report.json
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
REPORT = ROOT / "outputs" / "daily_from_god_sparse_report.json"


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
    for w in (3, 5, 8, 12):
        df[f"roll_min{w}"] = close.rolling(w, min_periods=w).min()
        df[f"roll_max{w}"] = close.rolling(w, min_periods=w).max()
        df[f"local_low{w}"] = close <= df[f"roll_min{w}"]
        df[f"local_high{w}"] = close >= df[f"roll_max{w}"]
    prev_ll = df["local_low5"].shift(1).fillna(False).astype(bool)
    df["bounce_up"] = prev_ll & (close > close.shift(1))
    prev_hh = df["local_high5"].shift(1).fillna(False).astype(bool)
    df["fade_down"] = prev_hh & (close < close.shift(1))
    # 更深的確認：前棒 8 低／高
    prev_ll8 = df["local_low8"].shift(1).fillna(False).astype(bool)
    df["bounce_up8"] = prev_ll8 & (close > close.shift(1))
    prev_hh8 = df["local_high8"].shift(1).fillna(False).astype(bool)
    df["fade_down8"] = prev_hh8 & (close < close.shift(1))

    day = df[df["session"] == "day"]
    night = df[df["session"] == "night"]
    day_open = day.groupby("date")["open"].first()
    day_close = day.groupby("date")["close"].last()
    dates = sorted(day_open.index.tolist())
    night_ret: dict[str, float] = {}
    for i, d in enumerate(dates):
        d_open_ts = day.loc[day["date"] == d, "ts"].iloc[0]
        if i == 0:
            nseg = night[night["ts"] < d_open_ts]
        else:
            prev = dates[i - 1]
            prev_last = day.loc[day["date"] == prev, "ts"].iloc[-1]
            nseg = night[(night["ts"] > prev_last) & (night["ts"] < d_open_ts)]
        night_ret[d] = (
            float(nseg["close"].iloc[-1] - nseg["open"].iloc[0]) if len(nseg) >= 2 else 0.0
        )
    df["night_ret"] = df["date"].map(night_ret).fillna(0.0)
    return df


def _session_blocks(df: pd.DataFrame) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
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
            blocks.append((i, j))
        i = j
    return blocks


def _throttle(
    mask: np.ndarray,
    blocks: list[tuple[int, int]],
    *,
    cooldown: int,
    max_per_block: int,
) -> np.ndarray:
    """每盤最多 N 筆；觸發後冷卻 cooldown 棒。"""
    out = np.zeros(len(mask), dtype=bool)
    for a, b in blocks:
        count = 0
        last = -10**9
        for i in range(a, b):
            if not bool(mask[i]):
                continue
            if i - last < int(cooldown):
                continue
            if count >= int(max_per_block):
                continue
            out[i] = True
            last = i
            count += 1
    return out


def _rule_catalog() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []

    def add(rid: str, side: str, desc: str, **kw: Any) -> None:
        rules.append({"id": rid, "side": side, "desc": desc, **kw})

    # —— 較稀多頭 ——
    add("L_bounce5", "long", "前棒5低＋收高", col="bounce_up")
    add(
        "L_bounce5_band",
        "long",
        "反彈5 + RSI14∈[35,50]",
        col="bounce_up",
        rsi14_lo=35.0,
        rsi14_hi=50.0,
    )
    add(
        "L_bounce5_band_ema",
        "long",
        "反彈5 + RSI帶 + <EMA20",
        col="bounce_up",
        rsi14_lo=35.0,
        rsi14_hi=50.0,
        below_ema20=True,
    )
    add("L_bounce8", "long", "前棒8低＋收高", col="bounce_up8")
    add(
        "L_bounce8_band",
        "long",
        "反彈8 + RSI14∈[32,48]",
        col="bounce_up8",
        rsi14_lo=32.0,
        rsi14_hi=48.0,
    )
    add(
        "L_ll8_band",
        "long",
        "8棒低 + RSI14∈[35,48]",
        col="local_low8",
        rsi14_lo=35.0,
        rsi14_hi=48.0,
    )
    add(
        "L_ll8_band_ema",
        "long",
        "8棒低 + RSI帶 + <EMA20",
        col="local_low8",
        rsi14_lo=35.0,
        rsi14_hi=48.0,
        below_ema20=True,
    )
    add(
        "L_ll12_band",
        "long",
        "12棒低 + RSI14∈[30,45]",
        col="local_low12",
        rsi14_lo=30.0,
        rsi14_hi=45.0,
    )
    add(
        "L_bounce5_early",
        "long",
        "反彈5 + RSI帶 + 日盤≤120分",
        col="bounce_up",
        rsi14_lo=35.0,
        rsi14_hi=50.0,
        max_mins_day=120,
    )
    add(
        "L_bounce5_night_ok",
        "long",
        "反彈5 + RSI帶 + 日盤夜報酬>-60",
        col="bounce_up",
        rsi14_lo=35.0,
        rsi14_hi=50.0,
        night_ret_gt=-60.0,
    )
    add(
        "L_ll8_below_ema60",
        "long",
        "8棒低 + <EMA60 + RSI14<45",
        col="local_low8",
        rsi14_hi=45.0,
        below_ema60=True,
    )

    # —— 較稀空頭 ——
    add("S_fade5", "short", "前棒5高＋收低", col="fade_down")
    add(
        "S_fade5_band",
        "short",
        "回落5 + RSI14∈[52,68]",
        col="fade_down",
        rsi14_lo=52.0,
        rsi14_hi=68.0,
    )
    add(
        "S_fade5_band_ema",
        "short",
        "回落5 + RSI帶 + >EMA20",
        col="fade_down",
        rsi14_lo=52.0,
        rsi14_hi=68.0,
        above_ema20=True,
    )
    add("S_fade8", "short", "前棒8高＋收低", col="fade_down8")
    add(
        "S_fade8_band",
        "short",
        "回落8 + RSI14∈[55,70]",
        col="fade_down8",
        rsi14_lo=55.0,
        rsi14_hi=70.0,
    )
    add(
        "S_lh8_band",
        "short",
        "8棒高 + RSI14∈[55,70]",
        col="local_high8",
        rsi14_lo=55.0,
        rsi14_hi=70.0,
    )
    add(
        "S_lh8_band_ema",
        "short",
        "8棒高 + RSI帶 + >EMA20",
        col="local_high8",
        rsi14_lo=55.0,
        rsi14_hi=70.0,
        above_ema20=True,
    )
    add(
        "S_lh12_band",
        "short",
        "12棒高 + RSI14∈[55,72]",
        col="local_high12",
        rsi14_lo=55.0,
        rsi14_hi=72.0,
    )
    add(
        "S_fade5_early",
        "short",
        "回落5 + RSI帶 + 日盤≤120分",
        col="fade_down",
        rsi14_lo=52.0,
        rsi14_hi=68.0,
        max_mins_day=120,
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
    if rule.get("below_ema60"):
        m &= ~df["above_ema60"].astype(bool).values
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


def _bt_long(df, mask, trading, slip, tp, stop, rid):
    return _long_only_tp_loop(
        df, _cfg(trading, slip, False, stop), mask, take_profit_points=tp, reason_entry=rid
    )


def _bt_short(df, mask, trading, slip, tp, stop, rid):
    z = np.zeros(len(df), dtype=bool)
    return _dual_tp_loop(
        df,
        _cfg(trading, slip, True, stop),
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


def _bt_dual(df, lm, sm, trading, slip, tp, stop, lid, sid):
    return _dual_tp_loop(
        df,
        _cfg(trading, slip, True, stop),
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


def _pass_edge(is_s, oos_s, *, min_dwr: float, min_cov: float) -> bool:
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
        and is_s["n_round_trips"] >= 20
        and oos_s["n_round_trips"] >= 10
    )


def main() -> int:
    p = argparse.ArgumentParser(description="sparse daily rules from god features")
    p.add_argument("--is-start", default="2024-08-01")
    p.add_argument("--is-end", default="2026-03-31")
    p.add_argument("--oos-start", default="2026-04-01")
    p.add_argument("--oos-end", default="2026-10-22")
    p.add_argument("--min-daily-win-rate", type=float, default=0.52)
    p.add_argument("--min-coverage", type=float, default=0.2)
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

    logger.info(f"{notify_prefix()} daily-sparse: prepare")
    is_df = _prepare(load_1m_range(args.is_start, args.is_end))
    oos_df = _prepare(load_1m_range(args.oos_start, args.oos_end))
    is_blocks = _session_blocks(is_df)
    oos_blocks = _session_blocks(oos_df)
    is_avail = _available(is_df)
    oos_avail = _available(oos_df)

    rules = _rule_catalog()
    # 節流：不強求多筆
    throttles = [
        {"cooldown": 30, "max_per_block": 1},
        {"cooldown": 60, "max_per_block": 1},
        {"cooldown": 15, "max_per_block": 1},
        {"cooldown": 30, "max_per_block": 2},
        {"cooldown": 90, "max_per_block": 1},
    ]
    # 稀訊號較可接受的 R:R（停損別遠大於停利太多）
    exit_grid = [
        (25.0, 40.0),
        (30.0, 50.0),
        (35.0, 55.0),
        (40.0, 60.0),
        (40.0, 80.0),
        (50.0, 80.0),
        (50.0, 100.0),
        (30.0, 80.0),
        (40.0, 100.0),
    ]

    total = len(rules) * len(throttles) * len(exit_grid)
    logger.info(
        f"{notify_prefix()} sparse grid rules={len(rules)} thr={len(throttles)} "
        f"exits={len(exit_grid)} total={total}"
    )

    bt_rows: list[dict[str, Any]] = []
    done = 0
    for rule in rules:
        raw_is = _eval_rule_mask(is_df, rule)
        raw_oos = _eval_rule_mask(oos_df, rule)
        for thr in throttles:
            mask_is = _throttle(
                raw_is, is_blocks, cooldown=thr["cooldown"], max_per_block=thr["max_per_block"]
            )
            mask_oos = _throttle(
                raw_oos, oos_blocks, cooldown=thr["cooldown"], max_per_block=thr["max_per_block"]
            )
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
                        "id": (
                            f"{rule['id']}_cd{thr['cooldown']}_mx{thr['max_per_block']}"
                            f"_tp{int(tp)}_st{int(stop)}"
                        ),
                        "rule": rule["id"],
                        "desc": rule["desc"],
                        "side": rule["side"],
                        "cooldown": thr["cooldown"],
                        "max_per_block": thr["max_per_block"],
                        "tp": tp,
                        "stop": stop,
                        "IS": is_s,
                        "OOS": oos_s,
                        "pass": ok,
                    }
                )
                done += 1
                if done % 80 == 0 or done == total:
                    logger.info(f"{notify_prefix()} sparse BT {done}/{total}")

    bt_rows.sort(
        key=lambda r: (
            r["pass"],
            r["IS"]["net_pnl"] > 0 and r["OOS"]["net_pnl"] > 0,
            r["OOS"]["net_pnl"],
            r["IS"]["net_pnl"],
        ),
        reverse=True,
    )

    # dual：同 thr／同 exit，IS&OOS 皆正的多空 top
    pos = [
        r
        for r in bt_rows
        if r["IS"]["net_pnl"] > 0 and r["OOS"]["net_pnl"] > 0
    ]
    longs = sorted(
        [r for r in pos if r["side"] == "long"],
        key=lambda r: r["OOS"]["net_pnl"],
        reverse=True,
    )[:6]
    shorts = sorted(
        [r for r in pos if r["side"] == "short"],
        key=lambda r: r["OOS"]["net_pnl"],
        reverse=True,
    )[:6]
    rule_by_id = {r["id"]: r for r in rules}
    dual_rows: list[dict[str, Any]] = []
    for L in longs:
        for S in shorts:
            if (
                L["tp"] != S["tp"]
                or L["stop"] != S["stop"]
                or L["cooldown"] != S["cooldown"]
                or L["max_per_block"] != S["max_per_block"]
            ):
                continue
            lr, sr = rule_by_id[L["rule"]], rule_by_id[S["rule"]]
            thr = {"cooldown": L["cooldown"], "max_per_block": L["max_per_block"]}
            tp, stop = float(L["tp"]), float(L["stop"])
            lm_is = _throttle(
                _eval_rule_mask(is_df, lr),
                is_blocks,
                cooldown=thr["cooldown"],
                max_per_block=thr["max_per_block"],
            )
            sm_is = _throttle(
                _eval_rule_mask(is_df, sr),
                is_blocks,
                cooldown=thr["cooldown"],
                max_per_block=thr["max_per_block"],
            )
            lm_oos = _throttle(
                _eval_rule_mask(oos_df, lr),
                oos_blocks,
                cooldown=thr["cooldown"],
                max_per_block=thr["max_per_block"],
            )
            sm_oos = _throttle(
                _eval_rule_mask(oos_df, sr),
                oos_blocks,
                cooldown=thr["cooldown"],
                max_per_block=thr["max_per_block"],
            )
            is_s = _bt_score(
                _bt_dual(is_df, lm_is, sm_is, trading, slip, tp, stop, lr["id"], sr["id"]),
                capital,
                is_avail,
            )
            oos_s = _bt_score(
                _bt_dual(oos_df, lm_oos, sm_oos, trading, slip, tp, stop, lr["id"], sr["id"]),
                capital,
                oos_avail,
            )
            ok = _pass_edge(is_s, oos_s, min_dwr=min_dwr, min_cov=min_cov)
            dual_rows.append(
                {
                    "id": (
                        f"{L['rule']}__{S['rule']}_cd{thr['cooldown']}_mx{thr['max_per_block']}"
                        f"_tp{int(tp)}_st{int(stop)}"
                    ),
                    "side": "dual",
                    "long": L["rule"],
                    "short": S["rule"],
                    "cooldown": thr["cooldown"],
                    "max_per_block": thr["max_per_block"],
                    "tp": tp,
                    "stop": stop,
                    "IS": is_s,
                    "OOS": oos_s,
                    "pass": ok,
                }
            )
    dual_rows.sort(
        key=lambda r: (r["pass"], r["OOS"]["net_pnl"], r["IS"]["net_pnl"]),
        reverse=True,
    )

    winners = [r for r in bt_rows + dual_rows if r["pass"]]
    pos_both = [r for r in bt_rows if r["IS"]["net_pnl"] > 0 and r["OOS"]["net_pnl"] > 0]

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "god_sparse_throttle_selective_BT",
        "exclude": ["rsi28_reclaim", "dense_local_every_bar"],
        "min_daily_win_rate": min_dwr,
        "min_coverage": min_cov,
        "goal": "正期望＋可定期做；不強求每日多筆",
        "n_pass": len(winners),
        "n_positive_is_oos": len(pos_both),
        "winners": winners[:15],
        "top15_positive": sorted(
            pos_both, key=lambda r: r["OOS"]["net_pnl"], reverse=True
        )[:15],
        "top15_single": bt_rows[:15],
        "top10_dual": dual_rows[:10],
        "note": (
            "節流＝冷卻＋每盤上限；規則偏 bounce／8–12棒＋RSI 中位帶。"
            "過關才考慮 tmf_daily_v1。"
        ),
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} sparse done pass={len(winners)} "
        f"pos_is_oos={len(pos_both)} -> {REPORT}"
    )
    if winners:
        w = winners[0]
        logger.info(
            f"  WINNER {w['id']} cov={w['OOS']['day_coverage']} "
            f"trips/d={w['OOS']['avg_trips_per_trade_day']} "
            f"IS={w['IS']['net_pnl']} OOS={w['OOS']['net_pnl']}"
        )
        return 0
    if pos_both:
        b = sorted(pos_both, key=lambda r: r["OOS"]["net_pnl"], reverse=True)[0]
        logger.info(
            f"  best_pos {b['id']} cov={b['OOS']['day_coverage']} "
            f"IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']} (gate fail)"
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
