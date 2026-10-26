from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.core.types import Side


@dataclass
class FastBreakoutConfig:
    lookback: int = 40
    capital: float = 10000
    point_value: float = 10
    fee_per_side: float = 20
    tax_rate: float = 0.00002
    slip_ticks: float = 1
    sessions: tuple[str, ...] = ("day",)
    # 控回撤：單筆浮虧達此點數即平倉；0=關閉。夜盤持倉也檢查（跳空風險）。
    stop_points: float = 0
    # 單日已實現虧損達此金額後，當日不再開新倉；0=關閉
    max_daily_loss: float = 0
    # 日盤最後一根強制平倉（不隔夜／不跨 session 留倉）
    flatten_eod: bool = False


def run_breakout_fast(bars: pd.DataFrame, cfg: FastBreakoutConfig) -> BacktestResult:
    """
    O(n) walk-forward breakout — no look-ahead:
    entry uses prior lookback window highs/lows (shifted).
    Research path; same economics as SecondBarBacktester.

    flatten_eod=True：允許 session 的最後一根 bar 強制平倉，不當沖隔夜。
    """
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)

    df = bars.reset_index(drop=True).copy()
    if df.empty:
        return BacktestResult(final_equity=cfg.capital)

    # Match SessionFilterSignal: off-session bars only HOLD (no entry, no exit check)
    # unless flatten_eod already closed on last in-session bar.
    # Lookback still uses full series including night.
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    ts = df["ts"].values if "ts" in df.columns else np.arange(len(df))
    if "session" in df.columns:
        sess = df["session"].astype(str).values
    else:
        sess = np.array(["day"] * len(df))
    allow = set(cfg.sessions) if cfg.sessions else {"day", "night"}
    n = len(df)
    in_sess = np.array([s in allow for s in sess], dtype=bool)
    # 允許 session 區塊的最後一根（下一根不在 allow，或資料結束）
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])

    prev_hi = pd.Series(high).rolling(cfg.lookback, min_periods=cfg.lookback).max().shift(1).values
    prev_lo = pd.Series(low).rolling(cfg.lookback, min_periods=cfg.lookback).min().shift(1).values

    equity = cfg.capital
    position = Side.FLAT
    entry = 0.0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    day_key = None
    day_realized = 0.0

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    def fill_buy(mid: float) -> float:
        return mid + cfg.slip_ticks

    def fill_sell(mid: float) -> float:
        return mid - cfg.slip_ticks

    def close_long(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, day_realized
        f = fill_sell(px)
        pts = f - entry
        pnl = pts * cfg.point_value
        c = cost(f)
        net = pnl - c
        equity += net
        day_realized += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0

    def close_short(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, day_realized
        f = fill_buy(px)
        pts = entry - f
        pnl = pts * cfg.point_value
        c = cost(f)
        net = pnl - c
        equity += net
        day_realized += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0

    for i in range(n):
        px = float(close[i])
        hi_ref = prev_hi[i]
        lo_ref = prev_lo[i]
        now = ts[i]
        in_session = bool(in_sess[i])
        eod = bool(cfg.flatten_eod and last_in_block[i])

        # calendar day for daily loss cap
        try:
            dk = str(pd.Timestamp(now).date())
        except Exception:
            dk = str(now)[:10]
        if dk != day_key:
            day_key = dk
            day_realized = 0.0

        # 停損：持倉時每根都查（含夜盤），優先於突破出場
        if position == Side.LONG and cfg.stop_points > 0 and (entry - px) >= cfg.stop_points:
            close_long(px, now, f"stop_loss_{cfg.stop_points}pt")
        elif position == Side.SHORT and cfg.stop_points > 0 and (px - entry) >= cfg.stop_points:
            close_short(px, now, f"stop_loss_{cfg.stop_points}pt")

        blocked_new = (
            cfg.max_daily_loss > 0 and day_realized <= -abs(cfg.max_daily_loss)
        )

        if in_session:
            # 日結最後一根：只平不開，避免開完立刻被 flatten
            if eod and position != Side.FLAT:
                if position == Side.LONG:
                    close_long(px, now, "flatten_eod")
                else:
                    close_short(px, now, "flatten_eod")
            elif not eod:
                if position == Side.FLAT and not blocked_new and not np.isnan(hi_ref):
                    if px > hi_ref:
                        f = fill_buy(px)
                        equity -= cost(f)
                        entry = f
                        position = Side.LONG
                        trades.append(
                            {"ts": now, "action": "BUY", "price": f, "reason": "breakout_fast"}
                        )
                    elif px < lo_ref:
                        f = fill_sell(px)
                        equity -= cost(f)
                        entry = f
                        position = Side.SHORT
                        trades.append(
                            {"ts": now, "action": "SELL", "price": f, "reason": "breakout_fast"}
                        )
                elif position == Side.LONG and not np.isnan(lo_ref) and px < lo_ref:
                    close_long(px, now, "breakout_exit")
                elif position == Side.SHORT and not np.isnan(hi_ref) and px > hi_ref:
                    close_short(px, now, "breakout_exit")

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
