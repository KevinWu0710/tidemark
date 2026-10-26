#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在鎖策略 RSI reclaim 進場上，加其他指標當「確認過濾」提高日勝率。
指標僅用 ≤t 資料（無偷看）。結果 → outputs/indicator_boost_report.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.backtest.engine import BacktestResult
from src.backtest.metrics import daily_pnl_stats, pair_round_trips, summarize
from src.core.config import load_settings, notify_prefix
from src.core.types import Side
from src.data.bars import load_1m_range


def _rsi(close: pd.Series, period: int) -> np.ndarray:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0).values


def _ema(close: pd.Series, span: int) -> np.ndarray:
    return close.ewm(span=span, adjust=False).mean().values


def _sma(close: pd.Series, window: int) -> np.ndarray:
    return close.rolling(window, min_periods=window).mean().values


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ef, es = _ema(close, fast), _ema(close, slow)
    line = ef - es
    sig = pd.Series(line).ewm(span=signal, adjust=False).mean().values
    hist = line - sig
    return line, sig, hist


def _stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    k = k.fillna(50.0)
    d = k.rolling(smooth, min_periods=smooth).mean().fillna(50.0)
    return k.values, d.values


def _bb(
    close: pd.Series, window: int = 20, nstd: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper = (mid + nstd * sd).values
    lower = (mid - nstd * sd).values
    return mid.values, upper, lower


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().values


def _adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(0).values


FilterFn = Callable[[int], bool]


def _build_filters(df: pd.DataFrame) -> dict[str, np.ndarray]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    n = len(df)
    macd_line, macd_sig, macd_hist = _macd(close)
    macd_hist_prev = np.roll(macd_hist, 1)
    macd_hist_prev[0] = np.nan
    macd_line_prev = np.roll(macd_line, 1)
    macd_line_prev[0] = np.nan
    k, d = _stoch(high, low, close, 14, 3)
    k_prev = np.roll(k, 1)
    k_prev[0] = np.nan
    bb_mid, bb_up, bb_lo = _bb(close, 20, 2.0)
    atr = _atr(df, 14)
    atr_ma = pd.Series(atr).rolling(60, min_periods=20).mean().values
    adx = _adx(df, 14)
    ema20, ema60 = _ema(close, 20), _ema(close, 60)
    sma20, sma60 = _sma(close, 20), _sma(close, 60)
    vol = df["volume"].astype(float).values if "volume" in df.columns else np.ones(n)
    vol_ma = pd.Series(vol).rolling(20, min_periods=5).mean().values

    masks: dict[str, np.ndarray] = {
        "none": np.ones(n, dtype=bool),
        "macd_hist_pos": macd_hist > 0,
        "macd_hist_up": (macd_hist > macd_hist_prev) & np.isfinite(macd_hist_prev),
        "macd_cross_up": (macd_line_prev <= macd_sig) & (macd_line > macd_sig) & np.isfinite(macd_line_prev),
        "macd_line_above_signal": macd_line > macd_sig,
        "stoch_k_gt_d": k > d,
        "stoch_reclaim_20": (k_prev < 20) & (k >= 20) & np.isfinite(k_prev),
        "stoch_not_overbought": k < 80,
        "bb_above_mid": close.values > bb_mid,
        "bb_bounce_lower": (close.values > bb_lo) & (close.shift(1).values <= bb_lo),
        "close_gt_ema20": close.values > ema20,
        "close_gt_ema60": close.values > ema60,
        "ema20_gt_ema60": ema20 > ema60,
        "close_gt_sma20": close.values > sma20,
        "close_gt_sma60": close.values > sma60,
        "atr_below_ma": atr < atr_ma,  # 波動未過熱
        "atr_above_ma": atr > atr_ma,  # 有波動
        "adx_trend_gt20": adx > 20,
        "adx_trend_gt25": adx > 25,
        "adx_chop_lt20": adx < 20,  # 震盪市（均值回歸較合）
        "vol_above_ma": vol > vol_ma,
    }
    # dual confirms
    masks["macd_hist_pos+close_gt_ema20"] = masks["macd_hist_pos"] & masks["close_gt_ema20"]
    masks["macd_hist_pos+stoch_k_gt_d"] = masks["macd_hist_pos"] & masks["stoch_k_gt_d"]
    masks["macd_line_above+ema20_gt_ema60"] = (
        masks["macd_line_above_signal"] & masks["ema20_gt_ema60"]
    )
    masks["adx_chop+macd_hist_up"] = masks["adx_chop_lt20"] & masks["macd_hist_up"]
    masks["adx_trend+macd_hist_pos"] = masks["adx_trend_gt20"] & masks["macd_hist_pos"]
    masks["bb_above_mid+macd_hist_pos"] = masks["bb_above_mid"] & masks["macd_hist_pos"]
    masks["stoch_reclaim+macd_hist_pos"] = masks["stoch_reclaim_20"] & masks["macd_hist_pos"]
    return masks


def run_rsi_with_mask(
    bars: pd.DataFrame,
    entry_ok: np.ndarray,
    *,
    period: int = 28,
    low_th: float = 20.0,
    take_profit: float = 40.0,
    stop_points: float = 300.0,
    capital: float = 10000.0,
) -> BacktestResult:
    df = bars.reset_index(drop=True)
    close = df["close"].astype(float)
    rsi = _rsi(close, period)
    r1 = np.roll(rsi, 1)
    r1[0] = np.nan
    raw = (r1 < low_th) & (rsi >= low_th)
    raw[: period + 2] = False
    if len(entry_ok) != len(df):
        raise ValueError("mask length mismatch")
    long_entry = raw & entry_ok

    sess = (
        df["session"].astype(str).values
        if "session" in df.columns
        else np.array(["day"] * len(df))
    )
    in_sess = np.array([s == "day" for s in sess], dtype=bool)
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])
    close_a = close.values
    ts = df["ts"].values if "ts" in df.columns else np.arange(len(df))
    n = len(df)
    equity = capital
    position = Side.FLAT
    entry = 0.0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    fee, tax, slip, pv = 20.0, 0.00002, 1.0, 10.0

    def cost(px: float) -> float:
        return fee + px * pv * tax

    def close_long(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry
        f = px - slip
        net = (f - entry) * pv - cost(f)
        equity += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0

    for i in range(n):
        px = float(close_a[i])
        now = ts[i]
        if position == Side.LONG:
            if (entry - px) >= stop_points:
                close_long(px, now, f"stop_loss_{stop_points}pt")
            elif (px - entry) >= take_profit:
                close_long(px, now, f"take_profit_{take_profit}pt")
        if in_sess[i] and last_in_block[i] and position == Side.LONG:
            close_long(px, now, "flatten_eod")
        if (
            in_sess[i]
            and not last_in_block[i]
            and position == Side.FLAT
            and bool(long_entry[i])
        ):
            f = px + slip
            equity -= cost(f)
            entry = f
            position = Side.LONG
            trades.append({"ts": now, "action": "BUY", "price": f, "reason": "rsi+filter"})
        mtm = equity + ((px - entry) * pv if position == Side.LONG else 0.0)
        if i % 10 == 0 or i == n - 1:
            curve.append({"ts": now, "equity": mtm, "price": px})

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        final_equity=curve[-1]["equity"] if curve else equity,
        n_steps=n,
    )


def _score(result: BacktestResult, capital: float) -> dict[str, Any]:
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


def main() -> int:
    settings = load_settings()
    capital = float(settings["trading"]["capital"])
    is_bars = load_1m_range("2024-08-01", "2026-03-31")
    oos_bars = load_1m_range("2026-04-01", "2026-10-22")
    is_masks = _build_filters(is_bars)
    oos_masks = _build_filters(oos_bars)

    report_path = ROOT / "outputs" / "indicator_boost_report.json"
    names = list(is_masks.keys())
    report_path.write_text(
        json.dumps({"status": "RUNNING", "n": len(names)}, indent=2), encoding="utf-8"
    )

    rows = []
    for name in names:
        is_r = run_rsi_with_mask(is_bars, is_masks[name], capital=capital)
        oos_r = run_rsi_with_mask(oos_bars, oos_masks[name], capital=capital)
        is_s, oos_s = _score(is_r, capital), _score(oos_r, capital)
        perfect = bool(
            (is_s.get("n_loss_days") or 0) == 0
            and (oos_s.get("n_loss_days") or 0) == 0
            and is_s["net_pnl"] > 0
            and oos_s["net_pnl"] > 0
            and (is_s.get("n_trade_days") or 0) >= 5
            and (oos_s.get("n_trade_days") or 0) >= 5
        )
        both_pos = is_s["net_pnl"] > 0 and oos_s["net_pnl"] > 0
        row = {
            "filter": name,
            "IS": is_s,
            "OOS": oos_s,
            "pass_perfect": perfect,
            "pass_both_pos": both_pos,
            "min_daily_wr": min(
                is_s.get("daily_win_rate") or 0, oos_s.get("daily_win_rate") or 0
            ),
        }
        rows.append(row)
        logger.info(
            f"{notify_prefix()} {name} dWR_is={is_s.get('daily_win_rate')} "
            f"dWR_oos={oos_s.get('daily_win_rate')} "
            f"IS={is_s['net_pnl']} OOS={oos_s['net_pnl']}"
        )

    rows.sort(
        key=lambda r: (
            r["pass_perfect"],
            r["min_daily_wr"],
            r["pass_both_pos"],
            r["OOS"]["net_pnl"],
            r["IS"]["net_pnl"],
        ),
        reverse=True,
    )
    baseline = next(r for r in rows if r["filter"] == "none")
    payload = {
        "status": "DONE",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "base": "RSI28 reclaim>20 + TP40 + stop300 + day flatten",
        "baseline": baseline,
        "n_filters": len(rows),
        "n_perfect": sum(1 for r in rows if r["pass_perfect"]),
        "beat_baseline_min_dwr": [
            r
            for r in rows
            if r["filter"] != "none"
            and r["min_daily_wr"] > (baseline["min_daily_wr"] or 0)
            and r["pass_both_pos"]
        ][:15],
        "top15": rows[:15],
        "note": (
            "進場仍為 RSI reclaim；下列為確認過濾。"
            "已內建：MACD／KD／布林／EMA／SMA／ATR／ADX／量能。"
            "perfect=IS&OOS 無虧損交易日且淨利>0、交易日>=5。"
        ),
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"{notify_prefix()} indicator boost done -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
