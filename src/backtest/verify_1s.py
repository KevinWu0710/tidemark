"""1m signal + 1s fill verification for locked rsi_tp (no look-ahead)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.backtest.fast_signals import FastSignalConfig
from src.core.types import Side


def _minute_key(ts: Any) -> pd.Timestamp:
    return pd.Timestamp(ts).floor("min")


def run_rsi_tp_1m_signal_1s_fill(
    bars_1m: pd.DataFrame,
    bars_1s: pd.DataFrame,
    period: int,
    low_th: float,
    cfg: FastSignalConfig,
    *,
    take_profit_points: float,
    min_hold_seconds: int = 0,
    entry_lag_minutes: int = 0,
) -> BacktestResult:
    """
    訊號只在 1m 收盤評估（與鎖策略 timeframe 一致）；
    持倉期間用 1s close 檢查停利／停損／flatten_eod。

    entry_lag_minutes=0：訊號分鐘最後一根 1s（≈1m close）。
    entry_lag_minutes=1：下一分鐘第一根 1s（看完收盤再下，較接近實盤）。
    min_hold_seconds：進場後 N 秒內不觸發停利／停損（flatten_eod 仍可）。
    """
    if bars_1m is None or bars_1m.empty or bars_1s is None or bars_1s.empty:
        return BacktestResult(final_equity=cfg.capital)

    m = bars_1m.reset_index(drop=True).copy()
    s = bars_1s.reset_index(drop=True).copy()
    m["ts"] = pd.to_datetime(m["ts"])
    s["ts"] = pd.to_datetime(s["ts"])
    m = m.sort_values("ts").reset_index(drop=True)
    s = s.sort_values("ts").reset_index(drop=True)

    close_m = m["close"].astype(float)
    delta = close_m.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50.0).values
    r0 = rsi
    r1 = np.roll(rsi, 1)
    r1[0] = np.nan
    long_entry = (r1 < low_th) & (r0 >= low_th)
    long_entry[: period + 2] = False

    if "session" in m.columns:
        sess_m = m["session"].astype(str).values
    else:
        sess_m = np.array(["day"] * len(m))
    allow = set(cfg.sessions) if cfg.sessions else {"day", "night"}
    in_sess_m = np.array([x in allow for x in sess_m], dtype=bool)

    lag = max(0, int(entry_lag_minutes))
    entry_minutes: set[pd.Timestamp] = set()
    for i in range(len(m)):
        if not bool(long_entry[i]) or not bool(in_sess_m[i]):
            continue
        is_last = i == len(m) - 1 or (
            bool(in_sess_m[i]) and not bool(in_sess_m[i + 1])
        )
        if cfg.flatten_eod and is_last:
            continue
        sig_mk = _minute_key(m["ts"].iloc[i])
        fill_mk = sig_mk + pd.Timedelta(minutes=lag)
        if cfg.flatten_eod and lag > 0:
            fill_mask = m["ts"].map(_minute_key) == fill_mk
            fill_rows = m.index[fill_mask]
            if len(fill_rows) > 0:
                fi = int(fill_rows[0])
                fill_last = fi == len(m) - 1 or (
                    bool(in_sess_m[fi]) and not bool(in_sess_m[fi + 1])
                )
                if fill_last:
                    continue
        entry_minutes.add(fill_mk)

    if "session" in s.columns:
        sess_s = s["session"].astype(str).values
    else:
        sess_s = np.array(["day"] * len(s))
    in_sess_s = np.array([x in allow for x in sess_s], dtype=bool)
    last_in_block = in_sess_s & np.concatenate([~in_sess_s[1:], np.array([True])])

    minute_keys = s["ts"].map(_minute_key)
    last_sec_of_minute: dict[pd.Timestamp, int] = {}
    first_sec_of_minute: dict[pd.Timestamp, int] = {}
    for i, mk in enumerate(minute_keys):
        last_sec_of_minute[mk] = i
        if mk not in first_sec_of_minute:
            first_sec_of_minute[mk] = i

    close_a = s["close"].astype(float).values
    ts = s["ts"].values
    n = len(s)
    equity = cfg.capital
    position = Side.FLAT
    entry = 0.0
    entry_ts: Any = None
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    day_key = None
    day_realized = 0.0
    tp = float(take_profit_points)
    min_hold = max(0, int(min_hold_seconds))

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    def fill_buy(mid: float) -> float:
        return mid + cfg.slip_ticks

    def fill_sell(mid: float) -> float:
        return mid - cfg.slip_ticks

    def close_long(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, entry_ts, day_realized
        f = fill_sell(px)
        net = (f - entry) * cfg.point_value - cost(f)
        equity += net
        day_realized += net
        trades.append(
            {"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason}
        )
        position = Side.FLAT
        entry = 0.0
        entry_ts = None

    def held_long_enough(now: Any) -> bool:
        if min_hold <= 0 or entry_ts is None:
            return True
        try:
            return (pd.Timestamp(now) - pd.Timestamp(entry_ts)).total_seconds() >= min_hold
        except Exception:
            return True

    for i in range(n):
        px = float(close_a[i])
        now = ts[i]
        mk = minute_keys.iloc[i]
        in_session = bool(in_sess_s[i])
        eod = bool(cfg.flatten_eod and last_in_block[i])
        try:
            dk = str(pd.Timestamp(now).date())
        except Exception:
            dk = str(now)[:10]
        if dk != day_key:
            day_key = dk
            day_realized = 0.0

        if position == Side.LONG and held_long_enough(now):
            if cfg.stop_points > 0 and (entry - px) >= cfg.stop_points:
                close_long(px, now, f"stop_loss_{cfg.stop_points}pt")
            elif tp > 0 and (px - entry) >= tp:
                close_long(px, now, f"take_profit_{tp}pt")

        blocked = cfg.max_daily_loss > 0 and day_realized <= -abs(cfg.max_daily_loss)

        if lag == 0:
            is_entry_bar = last_sec_of_minute.get(mk) == i
        else:
            is_entry_bar = first_sec_of_minute.get(mk) == i

        if in_session:
            if eod and position == Side.LONG:
                close_long(px, now, "flatten_eod")
            elif (
                not eod
                and position == Side.FLAT
                and not blocked
                and mk in entry_minutes
                and is_entry_bar
            ):
                f = fill_buy(px)
                equity -= cost(f)
                entry = f
                entry_ts = now
                position = Side.LONG
                trades.append(
                    {
                        "ts": now,
                        "action": "BUY",
                        "price": f,
                        "reason": "rsi_tp_long_1s_fill",
                    }
                )

        mtm = equity
        if position == Side.LONG:
            mtm = equity + (px - entry) * cfg.point_value
        if i % 60 == 0 or i == n - 1 or (trades and trades[-1]["ts"] == now):
            curve.append({"ts": now, "equity": mtm, "price": px})

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        final_equity=curve[-1]["equity"] if curve else equity,
        n_steps=n,
    )
