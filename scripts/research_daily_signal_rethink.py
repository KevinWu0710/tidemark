#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日頻訊號重想（放棄「局部高低 + 固定 TP40」賠錢結構）。

三族全新進場（皆因果、可節流為每盤 0～1 筆）：
  A) open_bias   — 開盤探針後依夜盤／缺口／探針方向做多或空一次
  B) squeeze_brk — 波動壓縮後突破（動能，非均值回歸）
  C) trend_pb    — 順勢（相對 EMA60）回踩 EMA20 再續攻／續空

出場：進場當下 ATR 倍數（非固定 40 點）；可加持倉棒數上限。
結果 → outputs/daily_signal_rethink_report.json
不過關不 enable tmf_daily_v1。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.metrics import (
    daily_pnl_stats,
    pair_round_trips,
    passes_session_daily_win_rate,
    summarize,
)
from src.backtest.engine import BacktestResult
from src.core.config import load_settings, notify_prefix
from src.core.types import Side
from src.data.bars import load_1m_range
from src.data.sessions import label_session

DAY_OPEN_MIN = 8 * 60 + 45
REPORT = ROOT / "outputs" / "daily_signal_rethink_report.json"


def _prepare(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.reset_index(drop=True).copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if "session" not in df.columns:
        df["session"] = df["ts"].map(label_session)
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["mins"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute - DAY_OPEN_MIN
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_c = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=14).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema60"] = close.ewm(span=60, adjust=False).mean()
    # squeeze: 短窗振幅 / 長窗振幅
    rng20 = high.rolling(20, min_periods=20).max() - low.rolling(20, min_periods=20).min()
    rng60 = high.rolling(60, min_periods=60).max() - low.rolling(60, min_periods=60).min()
    df["squeeze"] = rng20 / rng60.replace(0, np.nan)
    df["roll_hi20"] = high.rolling(20, min_periods=20).max().shift(1)
    df["roll_lo20"] = low.rolling(20, min_periods=20).min().shift(1)
    df["roll_hi12"] = high.rolling(12, min_periods=12).max().shift(1)
    df["roll_lo12"] = low.rolling(12, min_periods=12).min().shift(1)

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
        night_ret[d] = (
            float(nseg["close"].iloc[-1] - nseg["open"].iloc[0]) if len(nseg) >= 2 else 0.0
        )
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
        if j - i >= 15:
            blocks.append((str(d0), str(s0), i, j))
        i = j
    return blocks


def _masks_open_bias(
    df: pd.DataFrame,
    *,
    probe: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    每盤探針結束後進場一次。
    mode:
      cont_night — 夜盤同向續做（夜漲→做多，夜跌→做空）；日盤用 night_ret，夜盤用 gap
      fade_night — 反向
      probe_dir  — 跟探針區間方向（收在中位上／下）
      gap_cont   — 只日盤：缺口同向；夜盤 flat
    """
    n = len(df)
    long_m = np.zeros(n, dtype=bool)
    short_m = np.zeros(n, dtype=bool)
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    night_ret = df["night_ret"].astype(float).values
    gap = df["gap_pts"].astype(float).values
    for _d, sess, a, b in _session_blocks(df):
        if b - a <= probe + 1:
            continue
        i_ent = a + probe
        ph = float(np.max(high[a:i_ent]))
        pl = float(np.min(low[a:i_ent]))
        mid = 0.5 * (ph + pl)
        px = float(close[i_ent])
        nr = float(night_ret[i_ent])
        gp = float(gap[i_ent])
        side = None
        if mode == "cont_night":
            bias = nr if sess == "day" else gp
            if bias > 20:
                side = "long"
            elif bias < -20:
                side = "short"
        elif mode == "fade_night":
            bias = nr if sess == "day" else gp
            if bias > 40:
                side = "short"
            elif bias < -40:
                side = "long"
        elif mode == "probe_dir":
            if px > mid:
                side = "long"
            elif px < mid:
                side = "short"
        elif mode == "gap_cont":
            if sess != "day":
                continue
            if gp > 30:
                side = "long"
            elif gp < -30:
                side = "short"
        elif mode == "probe_with_night":
            bias = nr if sess == "day" else gp
            if px > mid and bias >= 0:
                side = "long"
            elif px < mid and bias <= 0:
                side = "short"
        if side == "long":
            long_m[i_ent] = True
        elif side == "short":
            short_m[i_ent] = True
    return long_m, short_m


def _masks_squeeze(
    df: pd.DataFrame,
    *,
    sq_th: float,
    once: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    close = df["close"].astype(float).values
    sq = df["squeeze"].astype(float).values
    hi = df["roll_hi20"].astype(float).values
    lo = df["roll_lo20"].astype(float).values
    sess = df["session"].astype(str).values
    long_m = np.zeros(n, dtype=bool)
    short_m = np.zeros(n, dtype=bool)
    raw_l = (sq < sq_th) & (close > hi) & np.isfinite(hi) & np.isin(sess, ["day", "night"])
    raw_s = (sq < sq_th) & (close < lo) & np.isfinite(lo) & np.isin(sess, ["day", "night"])
    if not once:
        return raw_l, raw_s
    for _d, _s, a, b in _session_blocks(df):
        fired = False
        for i in range(a, b):
            if fired:
                break
            if raw_l[i]:
                long_m[i] = True
                fired = True
            elif raw_s[i]:
                short_m[i] = True
                fired = True
    return long_m, short_m


def _masks_trend_pullback(
    df: pd.DataFrame,
    *,
    once: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """順勢回踩：多頭 close>ema60，曾觸及 ema20 帶後收上；空頭對稱。"""
    n = len(df)
    close = df["close"].astype(float).values
    low = df["low"].astype(float).values
    high = df["high"].astype(float).values
    e20 = df["ema20"].astype(float).values
    e60 = df["ema60"].astype(float).values
    sess = df["session"].astype(str).values
    long_m = np.zeros(n, dtype=bool)
    short_m = np.zeros(n, dtype=bool)
    # 前 3 棒曾碰 EMA20 附近（±0.15*ATR 用固定 8 點近似，後面用 ATR walk）
    for i in range(3, n):
        if sess[i] not in ("day", "night"):
            continue
        if not np.isfinite(e20[i]) or not np.isfinite(e60[i]):
            continue
        band = 8.0
        touched = any(abs(float(low[j]) - float(e20[j])) <= band for j in range(i - 3, i))
        touched_h = any(abs(float(high[j]) - float(e20[j])) <= band for j in range(i - 3, i))
        if close[i] > e60[i] and touched and close[i] > e20[i] and close[i] > close[i - 1]:
            long_m[i] = True
        if close[i] < e60[i] and touched_h and close[i] < e20[i] and close[i] < close[i - 1]:
            short_m[i] = True
    if not once:
        return long_m, short_m
    out_l = np.zeros(n, dtype=bool)
    out_s = np.zeros(n, dtype=bool)
    for _d, _s, a, b in _session_blocks(df):
        for i in range(a, b):
            if long_m[i]:
                out_l[i] = True
                break
            if short_m[i]:
                out_s[i] = True
                break
    return out_l, out_s


def _masks_or_break(
    df: pd.DataFrame,
    *,
    probe: int,
) -> tuple[np.ndarray, np.ndarray]:
    """每盤 opening range 突破一次（與舊 TORB 類似但日+夜、ATR 出場）。"""
    n = len(df)
    long_m = np.zeros(n, dtype=bool)
    short_m = np.zeros(n, dtype=bool)
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    for _d, _s, a, b in _session_blocks(df):
        if b - a <= probe + 2:
            continue
        ph = float(np.max(high[a : a + probe]))
        pl = float(np.min(low[a : a + probe]))
        fired = False
        for i in range(a + probe, b):
            if fired:
                break
            if close[i] > ph:
                long_m[i] = True
                fired = True
            elif close[i] < pl:
                short_m[i] = True
                fired = True
    return long_m, short_m


@dataclass
class WalkCfg:
    capital: float
    point_value: float
    fee_per_side: float
    tax_rate: float
    slip_ticks: float
    tp_atr: float
    stop_atr: float
    hold_bars: int  # 0 = no time stop
    flatten_eod: bool = True


def _atr_dual_walk(
    df: pd.DataFrame,
    long_entry: np.ndarray,
    short_entry: np.ndarray,
    cfg: WalkCfg,
    *,
    prefer: str = "long",
) -> BacktestResult:
    """一口；TP/stop = 進場 ATR × 倍數；可選持倉棒數上限；EOD 平。"""
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    atr = df["atr14"].astype(float).values
    ts = df["ts"].values
    sess = df["session"].astype(str).values
    n = len(df)
    in_sess = np.isin(sess, ["day", "night"])
    last_in = in_sess & np.concatenate([~in_sess[1:], np.array([True])])

    equity = cfg.capital
    position = Side.FLAT
    entry = 0.0
    entry_atr = 0.0
    bars_held = 0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    def fill_buy(mid: float) -> float:
        return mid + cfg.slip_ticks

    def fill_sell(mid: float) -> float:
        return mid - cfg.slip_ticks

    def close_long(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, entry_atr, bars_held
        f = fill_sell(px)
        net = (f - entry) * cfg.point_value - cost(f)
        equity += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0
        entry_atr = 0.0
        bars_held = 0

    def close_short(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, entry_atr, bars_held
        f = fill_buy(px)
        net = (entry - f) * cfg.point_value - cost(f)
        equity += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0
        entry_atr = 0.0
        bars_held = 0

    for i in range(n):
        px = float(close[i])
        now = ts[i]
        hj, lj = float(high[i]), float(low[i])
        eod = bool(cfg.flatten_eod and last_in[i])

        if position == Side.LONG:
            bars_held += 1
            tp = entry + cfg.tp_atr * entry_atr
            sl = entry - cfg.stop_atr * entry_atr
            if lj <= sl:
                close_long(sl, now, f"stop_{cfg.stop_atr}atr")
            elif hj >= tp:
                close_long(tp, now, f"tp_{cfg.tp_atr}atr")
            elif cfg.hold_bars > 0 and bars_held >= cfg.hold_bars:
                close_long(px, now, f"time_{cfg.hold_bars}")
        elif position == Side.SHORT:
            bars_held += 1
            tp = entry - cfg.tp_atr * entry_atr
            sl = entry + cfg.stop_atr * entry_atr
            if hj >= sl:
                close_short(sl, now, f"stop_{cfg.stop_atr}atr")
            elif lj <= tp:
                close_short(tp, now, f"tp_{cfg.tp_atr}atr")
            elif cfg.hold_bars > 0 and bars_held >= cfg.hold_bars:
                close_short(px, now, f"time_{cfg.hold_bars}")

        if in_sess[i]:
            if eod and position == Side.LONG:
                close_long(px, now, "flatten_eod")
            elif eod and position == Side.SHORT:
                close_short(px, now, "flatten_eod")
            elif not eod and position == Side.FLAT:
                want_l = bool(long_entry[i])
                want_s = bool(short_entry[i])
                a = float(atr[i]) if np.isfinite(atr[i]) and atr[i] > 0 else 0.0
                if a <= 0:
                    want_l = want_s = False
                go = None
                if want_l and want_s:
                    go = prefer
                elif want_l:
                    go = "long"
                elif want_s:
                    go = "short"
                if go == "long":
                    f = fill_buy(px)
                    equity -= cost(f)
                    entry = f
                    entry_atr = a
                    bars_held = 0
                    position = Side.LONG
                    trades.append({"ts": now, "action": "BUY", "price": f, "reason": "entry_long"})
                elif go == "short":
                    f = fill_sell(px)
                    equity -= cost(f)
                    entry = f
                    entry_atr = a
                    bars_held = 0
                    position = Side.SHORT
                    trades.append({"ts": now, "action": "SELL", "price": f, "reason": "entry_short"})

        mtm = equity
        if position == Side.LONG:
            mtm = equity + (px - entry) * cfg.point_value
        elif position == Side.SHORT:
            mtm = equity + (entry - px) * cfg.point_value
        curve.append({"ts": now, "equity": mtm, "price": px})

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        final_equity=curve[-1]["equity"] if curve else equity,
        n_steps=len(curve),
    )


def _available(bars: pd.DataFrame) -> set[str]:
    df = bars
    if "session" in df.columns:
        df = df[df["session"].astype(str).isin(["day", "night"])]
    return set(pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d").tolist())


def _bt_score(result: BacktestResult, capital: float, available: set[str]) -> dict[str, Any]:
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


def _pass(is_s, oos_s, *, min_dwr: float, min_cov: float) -> bool:
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


def _recipe_catalog() -> list[dict[str, Any]]:
    rec: list[dict[str, Any]] = []
    for probe in (5, 10, 15, 30):
        for mode in ("cont_night", "fade_night", "probe_dir", "probe_with_night", "gap_cont"):
            rec.append(
                {
                    "id": f"open_{mode}_p{probe}",
                    "family": "open_bias",
                    "desc": f"開盤探針{probe}分／{mode}",
                    "kind": "open",
                    "probe": probe,
                    "mode": mode,
                }
            )
    for th in (0.35, 0.45, 0.55):
        for once in (True, False):
            rec.append(
                {
                    "id": f"sq_{th}_once{int(once)}",
                    "family": "squeeze_brk",
                    "desc": f"壓縮<{th}後突破 once={once}",
                    "kind": "squeeze",
                    "sq_th": th,
                    "once": once,
                }
            )
    for once in (True, False):
        rec.append(
            {
                "id": f"trend_pb_once{int(once)}",
                "family": "trend_pb",
                "desc": f"EMA60順勢回踩EMA20 once={once}",
                "kind": "trend_pb",
                "once": once,
            }
        )
    for probe in (10, 15, 30):
        rec.append(
            {
                "id": f"or_brk_p{probe}",
                "family": "or_break",
                "desc": f"每盤OR{probe}棒突破一次",
                "kind": "or",
                "probe": probe,
            }
        )
    return rec


def _build_masks(df: pd.DataFrame, r: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    k = r["kind"]
    if k == "open":
        return _masks_open_bias(df, probe=int(r["probe"]), mode=str(r["mode"]))
    if k == "squeeze":
        return _masks_squeeze(df, sq_th=float(r["sq_th"]), once=bool(r["once"]))
    if k == "trend_pb":
        return _masks_trend_pullback(df, once=bool(r["once"]))
    if k == "or":
        return _masks_or_break(df, probe=int(r["probe"]))
    raise ValueError(k)


def main() -> int:
    p = argparse.ArgumentParser(description="rethink daily signals (ATR exits, new families)")
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
    base = dict(
        capital=capital,
        point_value=float(trading["point_value"]),
        fee_per_side=float(trading.get("fee_per_side", 20)),
        tax_rate=float(trading.get("tax_rate", 0.00002)),
        slip_ticks=slip,
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {"status": "RUNNING", "updated_at": datetime.now().isoformat(timespec="seconds")},
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(f"{notify_prefix()} signal-rethink: prepare (no TP40 labels)")
    is_df = _prepare(load_1m_range(args.is_start, args.is_end))
    oos_df = _prepare(load_1m_range(args.oos_start, args.oos_end))
    is_avail = _available(is_df)
    oos_avail = _available(oos_df)
    med_atr = float(is_df["atr14"].median())
    logger.info(f"{notify_prefix()} IS median ATR14≈{med_atr:.2f}")

    recipes = _recipe_catalog()
    # ATR 倍數出場（刻意不含固定 40 點）
    exits = [
        # (tp_atr, stop_atr, hold_bars)
        (1.0, 1.0, 0),
        (1.5, 1.0, 0),
        (2.0, 1.0, 0),
        (1.5, 1.5, 0),
        (2.0, 1.5, 0),
        (1.0, 1.5, 0),
        (1.5, 1.0, 30),
        (2.0, 1.5, 60),
        (1.0, 1.0, 20),
    ]

    total = len(recipes) * len(exits)
    logger.info(f"{notify_prefix()} recipes={len(recipes)} exits={len(exits)} total={total}")

    rows: list[dict[str, Any]] = []
    done = 0
    for r in recipes:
        lm_is, sm_is = _build_masks(is_df, r)
        lm_oos, sm_oos = _build_masks(oos_df, r)
        for tp_a, st_a, hold in exits:
            wcfg = WalkCfg(
                **base,
                tp_atr=tp_a,
                stop_atr=st_a,
                hold_bars=hold,
            )
            is_s = _bt_score(
                _atr_dual_walk(is_df, lm_is, sm_is, wcfg), capital, is_avail
            )
            oos_s = _bt_score(
                _atr_dual_walk(oos_df, lm_oos, sm_oos, wcfg), capital, oos_avail
            )
            ok = _pass(is_s, oos_s, min_dwr=min_dwr, min_cov=min_cov)
            rows.append(
                {
                    "id": f"{r['id']}_tpA{tp_a}_stA{st_a}_h{hold}",
                    "family": r["family"],
                    "desc": r["desc"],
                    "tp_atr": tp_a,
                    "stop_atr": st_a,
                    "hold_bars": hold,
                    "approx_tp_pts": round(med_atr * tp_a, 1),
                    "approx_stop_pts": round(med_atr * st_a, 1),
                    "IS": is_s,
                    "OOS": oos_s,
                    "pass": ok,
                }
            )
            done += 1
            if done % 50 == 0 or done == total:
                logger.info(f"{notify_prefix()} rethink BT {done}/{total}")

    rows.sort(
        key=lambda x: (
            x["pass"],
            x["IS"]["net_pnl"] > 0 and x["OOS"]["net_pnl"] > 0,
            x["OOS"]["net_pnl"],
            x["IS"]["net_pnl"],
        ),
        reverse=True,
    )
    winners = [x for x in rows if x["pass"]]
    pos = [x for x in rows if x["IS"]["net_pnl"] > 0 and x["OOS"]["net_pnl"] > 0]
    by_fam: dict[str, list] = {}
    for x in rows:
        by_fam.setdefault(x["family"], []).append(x)
    fam_best = {
        f: sorted(
            xs,
            key=lambda z: (z["OOS"]["net_pnl"], z["IS"]["net_pnl"]),
            reverse=True,
        )[0]
        for f, xs in by_fam.items()
        if xs
    }

    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "rethink_open_squeeze_trend_ATR_exits",
        "abandoned": [
            "local_low_mean_reversion_primary",
            "fixed_tp40_label_and_exit",
            "rsi28_reclaim",
        ],
        "is_median_atr14": round(med_atr, 2),
        "min_daily_win_rate": min_dwr,
        "min_coverage": min_cov,
        "n_pass": len(winners),
        "n_positive_is_oos": len(pos),
        "winners": winners[:15],
        "top15_positive": sorted(pos, key=lambda z: z["OOS"]["net_pnl"], reverse=True)[:15],
        "top15_all": rows[:15],
        "best_per_family": {k: v for k, v in fam_best.items()},
        "note": "新訊號族 + ATR 倍數出場；過關才考慮 tmf_daily_v1。",
    }
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        f"{notify_prefix()} rethink done pass={len(winners)} pos={len(pos)} -> {REPORT}"
    )
    if winners:
        w = winners[0]
        logger.info(
            f"  WINNER {w['id']} IS={w['IS']['net_pnl']} OOS={w['OOS']['net_pnl']} "
            f"cov={w['OOS']['day_coverage']}"
        )
        return 0
    if pos:
        b = sorted(pos, key=lambda z: z["OOS"]["net_pnl"], reverse=True)[0]
        logger.info(
            f"  best_pos {b['id']} IS={b['IS']['net_pnl']} OOS={b['OOS']['net_pnl']}"
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
