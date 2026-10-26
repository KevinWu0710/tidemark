from __future__ import annotations

"""
O(n) walk-forward runners for research scan (ema / rsi / sma_cross / breakout long-only).
Same session / stop / flatten_eod / cost rules as fast_breakout.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.core.types import Side


@dataclass
class FastSignalConfig:
    capital: float = 10000
    point_value: float = 10
    fee_per_side: float = 20
    tax_rate: float = 0.00002
    slip_ticks: float = 1
    sessions: tuple[str, ...] = ("day",)
    stop_points: float = 0
    max_daily_loss: float = 0
    flatten_eod: bool = False
    allow_short: bool = True
    # strategy-specific (unused by runner itself)
    meta: dict[str, Any] | None = None


def _run_from_masks(
    bars: pd.DataFrame,
    cfg: FastSignalConfig,
    *,
    long_entry: np.ndarray,
    short_entry: np.ndarray,
    long_exit: np.ndarray,
    short_exit: np.ndarray,
    reason_prefix: str,
) -> BacktestResult:
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)

    df = bars.reset_index(drop=True)
    close = df["close"].astype(float).values
    ts = df["ts"].values if "ts" in df.columns else np.arange(len(df))
    if "session" in df.columns:
        sess = df["session"].astype(str).values
    else:
        sess = np.array(["day"] * len(df))
    allow = set(cfg.sessions) if cfg.sessions else {"day", "night"}
    n = len(df)
    in_sess = np.array([s in allow for s in sess], dtype=bool)
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])

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
        net = (f - entry) * cfg.point_value - cost(f)
        equity += net
        day_realized += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0

    def close_short(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, day_realized
        f = fill_buy(px)
        net = (entry - f) * cfg.point_value - cost(f)
        equity += net
        day_realized += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0

    for i in range(n):
        px = float(close[i])
        now = ts[i]
        in_session = bool(in_sess[i])
        eod = bool(cfg.flatten_eod and last_in_block[i])

        try:
            dk = str(pd.Timestamp(now).date())
        except Exception:
            dk = str(now)[:10]
        if dk != day_key:
            day_key = dk
            day_realized = 0.0

        if position == Side.LONG and cfg.stop_points > 0 and (entry - px) >= cfg.stop_points:
            close_long(px, now, f"stop_loss_{cfg.stop_points}pt")
        elif position == Side.SHORT and cfg.stop_points > 0 and (px - entry) >= cfg.stop_points:
            close_short(px, now, f"stop_loss_{cfg.stop_points}pt")

        blocked = cfg.max_daily_loss > 0 and day_realized <= -abs(cfg.max_daily_loss)

        if in_session:
            if eod and position != Side.FLAT:
                if position == Side.LONG:
                    close_long(px, now, "flatten_eod")
                else:
                    close_short(px, now, "flatten_eod")
            elif not eod:
                if position == Side.FLAT and not blocked:
                    if bool(long_entry[i]):
                        f = fill_buy(px)
                        equity -= cost(f)
                        entry = f
                        position = Side.LONG
                        trades.append(
                            {
                                "ts": now,
                                "action": "BUY",
                                "price": f,
                                "reason": f"{reason_prefix}_long",
                            }
                        )
                    elif cfg.allow_short and bool(short_entry[i]):
                        f = fill_sell(px)
                        equity -= cost(f)
                        entry = f
                        position = Side.SHORT
                        trades.append(
                            {
                                "ts": now,
                                "action": "SELL",
                                "price": f,
                                "reason": f"{reason_prefix}_short",
                            }
                        )
                elif position == Side.LONG and bool(long_exit[i]):
                    close_long(px, now, f"{reason_prefix}_exit")
                elif position == Side.SHORT and bool(short_exit[i]):
                    close_short(px, now, f"{reason_prefix}_exit")

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


def run_ema_fast(bars: pd.DataFrame, period: int, cfg: FastSignalConfig) -> BacktestResult:
    close = bars["close"].astype(float)
    ema = close.ewm(span=period, adjust=False).mean()
    c = close.values
    e = ema.values
    e_prev = np.roll(e, 1)
    e_prev[0] = np.nan
    rising = e > e_prev
    falling = e < e_prev
    long_entry = (c > e) & rising
    short_entry = (c < e) & falling
    long_exit = c < e
    short_exit = c > e
    # warmup
    long_entry[: period + 1] = False
    short_entry[: period + 1] = False
    return _run_from_masks(
        bars,
        cfg,
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=long_exit,
        short_exit=short_exit,
        reason_prefix="ema_mom",
    )


def _long_only_tp_loop(
    bars: pd.DataFrame,
    cfg: FastSignalConfig,
    long_entry: np.ndarray,
    *,
    take_profit_points: float,
    reason_entry: str,
) -> BacktestResult:
    """Shared long-only TP/stop/flatten walker (RSI-tp / MACD-tp)."""
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)

    df = bars.reset_index(drop=True)
    close_a = df["close"].astype(float).values
    ts = df["ts"].values if "ts" in df.columns else np.arange(len(df))
    if "session" in df.columns:
        sess = df["session"].astype(str).values
    else:
        sess = np.array(["day"] * len(df))
    allow = set(cfg.sessions) if cfg.sessions else {"day", "night"}
    n = len(df)
    in_sess = np.array([s in allow for s in sess], dtype=bool)
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])
    if len(long_entry) != n:
        raise ValueError("long_entry length mismatch")

    equity = cfg.capital
    position = Side.FLAT
    entry = 0.0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    day_key = None
    day_realized = 0.0
    tp = float(take_profit_points)

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    def fill_buy(mid: float) -> float:
        return mid + cfg.slip_ticks

    def fill_sell(mid: float) -> float:
        return mid - cfg.slip_ticks

    def close_long(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, day_realized
        f = fill_sell(px)
        net = (f - entry) * cfg.point_value - cost(f)
        equity += net
        day_realized += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0

    for i in range(n):
        px = float(close_a[i])
        now = ts[i]
        in_session = bool(in_sess[i])
        eod = bool(cfg.flatten_eod and last_in_block[i])
        try:
            dk = str(pd.Timestamp(now).date())
        except Exception:
            dk = str(now)[:10]
        if dk != day_key:
            day_key = dk
            day_realized = 0.0

        if position == Side.LONG:
            if cfg.stop_points > 0 and (entry - px) >= cfg.stop_points:
                close_long(px, now, f"stop_loss_{cfg.stop_points}pt")
            elif tp > 0 and (px - entry) >= tp:
                close_long(px, now, f"take_profit_{tp}pt")

        blocked = cfg.max_daily_loss > 0 and day_realized <= -abs(cfg.max_daily_loss)

        if in_session:
            if eod and position == Side.LONG:
                close_long(px, now, "flatten_eod")
            elif not eod and position == Side.FLAT and not blocked and bool(long_entry[i]):
                f = fill_buy(px)
                equity -= cost(f)
                entry = f
                position = Side.LONG
                trades.append(
                    {"ts": now, "action": "BUY", "price": f, "reason": reason_entry}
                )

        mtm = equity
        if position == Side.LONG:
            mtm = equity + (px - entry) * cfg.point_value
        curve.append({"ts": now, "equity": mtm, "price": px})

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        final_equity=curve[-1]["equity"] if curve else equity,
        n_steps=len(curve),
    )


def _dual_tp_loop(
    bars: pd.DataFrame,
    cfg: FastSignalConfig,
    long_entry: np.ndarray,
    short_entry: np.ndarray,
    *,
    take_profit_long: float,
    take_profit_short: float,
    stop_long: float,
    stop_short: float,
    reason_long: str,
    reason_short: str,
    prefer: str = "long",
) -> BacktestResult:
    """
    多訊號／空訊號分開進場，策略層一口：有倉時忽略反向進場。
    同一根若多空同時觸發：prefer=long|short。
    """
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)

    df = bars.reset_index(drop=True)
    close_a = df["close"].astype(float).values
    ts = df["ts"].values if "ts" in df.columns else np.arange(len(df))
    if "session" in df.columns:
        sess = df["session"].astype(str).values
    else:
        sess = np.array(["day"] * len(df))
    allow = set(cfg.sessions) if cfg.sessions else {"day", "night"}
    n = len(df)
    in_sess = np.array([s in allow for s in sess], dtype=bool)
    last_in_block = in_sess & np.concatenate([~in_sess[1:], np.array([True])])
    if len(long_entry) != n or len(short_entry) != n:
        raise ValueError("entry mask length mismatch")

    equity = cfg.capital
    position = Side.FLAT
    entry = 0.0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    day_key = None
    day_realized = 0.0
    tp_l = float(take_profit_long)
    tp_s = float(take_profit_short)
    st_l = float(stop_long)
    st_s = float(stop_short)
    prefer_long = str(prefer).lower() != "short"

    def cost(px: float) -> float:
        return cfg.fee_per_side + px * cfg.point_value * cfg.tax_rate

    def fill_buy(mid: float) -> float:
        return mid + cfg.slip_ticks

    def fill_sell(mid: float) -> float:
        return mid - cfg.slip_ticks

    def close_long(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, day_realized
        f = fill_sell(px)
        net = (f - entry) * cfg.point_value - cost(f)
        equity += net
        day_realized += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0

    def close_short(px: float, now: Any, reason: str) -> None:
        nonlocal equity, position, entry, day_realized
        f = fill_buy(px)
        net = (entry - f) * cfg.point_value - cost(f)
        equity += net
        day_realized += net
        trades.append({"ts": now, "action": "CLOSE", "price": f, "pnl": net, "reason": reason})
        position = Side.FLAT
        entry = 0.0

    for i in range(n):
        px = float(close_a[i])
        now = ts[i]
        in_session = bool(in_sess[i])
        eod = bool(cfg.flatten_eod and last_in_block[i])
        try:
            dk = str(pd.Timestamp(now).date())
        except Exception:
            dk = str(now)[:10]
        if dk != day_key:
            day_key = dk
            day_realized = 0.0

        if position == Side.LONG:
            if st_l > 0 and (entry - px) >= st_l:
                close_long(px, now, f"stop_loss_{st_l}pt")
            elif tp_l > 0 and (px - entry) >= tp_l:
                close_long(px, now, f"take_profit_{tp_l}pt")
        elif position == Side.SHORT:
            if st_s > 0 and (px - entry) >= st_s:
                close_short(px, now, f"stop_loss_{st_s}pt")
            elif tp_s > 0 and (entry - px) >= tp_s:
                close_short(px, now, f"take_profit_{tp_s}pt")

        blocked = cfg.max_daily_loss > 0 and day_realized <= -abs(cfg.max_daily_loss)

        if in_session:
            if eod and position == Side.LONG:
                close_long(px, now, "flatten_eod")
            elif eod and position == Side.SHORT:
                close_short(px, now, "flatten_eod")
            elif not eod and position == Side.FLAT and not blocked:
                want_l = bool(long_entry[i])
                want_s = bool(short_entry[i]) and bool(cfg.allow_short)
                take_l = want_l and (not want_s or prefer_long)
                take_s = want_s and (not want_l or not prefer_long)
                if take_l:
                    f = fill_buy(px)
                    equity -= cost(f)
                    entry = f
                    position = Side.LONG
                    trades.append(
                        {"ts": now, "action": "BUY", "price": f, "reason": reason_long}
                    )
                elif take_s:
                    f = fill_sell(px)
                    equity -= cost(f)
                    entry = f
                    position = Side.SHORT
                    trades.append(
                        {"ts": now, "action": "SELL", "price": f, "reason": reason_short}
                    )

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


def _rsi_series(close: pd.Series, period: int) -> np.ndarray:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0).values


def rsi_tp_long_entry_mask(bars: pd.DataFrame, period: int, low_th: float) -> np.ndarray:
    close = bars["close"].astype(float)
    rsi = _rsi_series(close, period)
    r1 = np.roll(rsi, 1)
    r1[0] = np.nan
    long_entry = (r1 < low_th) & (rsi >= low_th)
    long_entry[: period + 2] = False
    return long_entry


def rsi_tp_short_entry_mask(bars: pd.DataFrame, period: int, high_th: float) -> np.ndarray:
    """超買回落：前一根 > high、本根 ≤ high → 做空。"""
    close = bars["close"].astype(float)
    rsi = _rsi_series(close, period)
    r1 = np.roll(rsi, 1)
    r1[0] = np.nan
    short_entry = (r1 > high_th) & (rsi <= high_th)
    short_entry[: period + 2] = False
    return short_entry


def run_rsi_tp_fast(
    bars: pd.DataFrame,
    period: int,
    low_th: float,
    cfg: FastSignalConfig,
    *,
    take_profit_points: float,
) -> BacktestResult:
    """
    高勝率取向：RSI 超賣回升只做多；達 take_profit_points 或停損／日結平倉。
    （小停利 + 相對寬停損 → 理論上較易拉高勝率）
    """
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)
    long_entry = rsi_tp_long_entry_mask(bars, period, low_th)
    return _long_only_tp_loop(
        bars, cfg, long_entry, take_profit_points=take_profit_points, reason_entry="rsi_tp_long"
    )


def run_rsi_tp_short_fast(
    bars: pd.DataFrame,
    period: int,
    high_th: float,
    cfg: FastSignalConfig,
    *,
    take_profit_points: float,
) -> BacktestResult:
    """RSI 超買回落只做空；小停利／停損／日結。"""
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)
    short_entry = rsi_tp_short_entry_mask(bars, period, high_th)
    z = np.zeros(len(bars), dtype=bool)
    cfg2 = FastSignalConfig(
        capital=cfg.capital,
        point_value=cfg.point_value,
        fee_per_side=cfg.fee_per_side,
        tax_rate=cfg.tax_rate,
        slip_ticks=cfg.slip_ticks,
        sessions=cfg.sessions,
        stop_points=cfg.stop_points,
        max_daily_loss=cfg.max_daily_loss,
        flatten_eod=cfg.flatten_eod,
        allow_short=True,
    )
    return _dual_tp_loop(
        bars,
        cfg2,
        z,
        short_entry,
        take_profit_long=0.0,
        take_profit_short=float(take_profit_points),
        stop_long=0.0,
        stop_short=float(cfg.stop_points),
        reason_long="unused",
        reason_short="rsi_tp_short",
        prefer="short",
    )


def run_dual_rsi_tp_fast(
    bars: pd.DataFrame,
    cfg: FastSignalConfig,
    *,
    long_period: int,
    long_low: float,
    long_tp: float,
    long_stop: float,
    short_period: int,
    short_high: float,
    short_tp: float,
    short_stop: float,
    prefer: str = "long",
) -> BacktestResult:
    """獨立多／空 RSI-TP 訊號合成（一口）。"""
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)
    long_entry = rsi_tp_long_entry_mask(bars, long_period, long_low)
    short_entry = rsi_tp_short_entry_mask(bars, short_period, short_high)
    cfg2 = FastSignalConfig(
        capital=cfg.capital,
        point_value=cfg.point_value,
        fee_per_side=cfg.fee_per_side,
        tax_rate=cfg.tax_rate,
        slip_ticks=cfg.slip_ticks,
        sessions=cfg.sessions,
        stop_points=0.0,
        max_daily_loss=cfg.max_daily_loss,
        flatten_eod=cfg.flatten_eod,
        allow_short=True,
    )
    return _dual_tp_loop(
        bars,
        cfg2,
        long_entry,
        short_entry,
        take_profit_long=float(long_tp),
        take_profit_short=float(short_tp),
        stop_long=float(long_stop),
        stop_short=float(short_stop),
        reason_long="rsi_tp_long",
        reason_short="rsi_tp_short",
        prefer=prefer,
    )


def run_macd_tp_fast(
    bars: pd.DataFrame,
    cfg: FastSignalConfig,
    *,
    take_profit_points: float,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    mode: str = "hist_cross_up",
) -> BacktestResult:
    """
    MACD 主訊號只做多 + 固定停利／停損／日結。
    mode:
      hist_cross_up — 柱由 ≤0 翻 >0
      line_cross_up — MACD 上穿 signal
      hist_up — 柱較前一根上升（較密）
      hist_pos — 柱 > 0（狀態，非交叉；進場後需靠出場）
    """
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)

    df = bars.reset_index(drop=True)
    close = df["close"].astype(float)
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_f - ema_s).values
    macd_sig = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    hist = macd_line - macd_sig
    h1 = np.roll(hist, 1)
    h1[0] = np.nan
    l1 = np.roll(macd_line, 1)
    l1[0] = np.nan
    s1 = np.roll(macd_sig, 1)
    s1[0] = np.nan

    warmup = slow + signal + 2
    if mode == "hist_cross_up":
        long_entry = (h1 <= 0) & (hist > 0) & np.isfinite(h1)
    elif mode == "line_cross_up":
        long_entry = (l1 <= s1) & (macd_line > macd_sig) & np.isfinite(l1)
    elif mode == "hist_up":
        long_entry = (hist > h1) & np.isfinite(h1)
    elif mode == "hist_pos":
        # edge: first bar hist becomes positive while flat — use cross for practical entry
        long_entry = (h1 <= 0) & (hist > 0) & np.isfinite(h1)
    else:
        raise ValueError(f"unknown macd mode: {mode}")
    long_entry[:warmup] = False

    return _long_only_tp_loop(
        bars,
        cfg,
        long_entry,
        take_profit_points=take_profit_points,
        reason_entry=f"macd_{mode}",
    )


def run_rsi_fast(
    bars: pd.DataFrame,
    period: int,
    low_th: float,
    high_th: float,
    cfg: FastSignalConfig,
) -> BacktestResult:
    close = bars["close"].astype(float)
    delta = close.diff()
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
    short_entry = (r1 > high_th) & (r0 <= high_th)
    long_exit = r0 >= high_th
    short_exit = r0 <= low_th
    long_entry[: period + 2] = False
    short_entry[: period + 2] = False
    return _run_from_masks(
        bars,
        cfg,
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=long_exit,
        short_exit=short_exit,
        reason_prefix="rsi_mr",
    )


def run_sma_cross_fast(
    bars: pd.DataFrame, fast: int, slow: int, cfg: FastSignalConfig
) -> BacktestResult:
    close = bars["close"].astype(float)
    f = close.rolling(fast, min_periods=fast).mean().values
    s = close.rolling(slow, min_periods=slow).mean().values
    f_prev = np.roll(f, 1)
    s_prev = np.roll(s, 1)
    f_prev[0] = np.nan
    s_prev[0] = np.nan
    # cross up / down using prior relation + current
    long_entry = (f_prev <= s_prev) & (f > s)
    short_entry = (f_prev >= s_prev) & (f < s)
    long_exit = f < s
    short_exit = f > s
    warm = slow + 1
    long_entry[:warm] = False
    short_entry[:warm] = False
    return _run_from_masks(
        bars,
        cfg,
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=long_exit,
        short_exit=short_exit,
        reason_prefix="sma_cross",
    )


def run_breakout_long_fast(
    bars: pd.DataFrame, lookback: int, cfg: FastSignalConfig
) -> BacktestResult:
    """Donchian long-only: break high to enter; break low to exit."""
    high = bars["high"].astype(float).values
    low = bars["low"].astype(float).values
    close = bars["close"].astype(float).values
    prev_hi = pd.Series(high).rolling(lookback, min_periods=lookback).max().shift(1).values
    prev_lo = pd.Series(low).rolling(lookback, min_periods=lookback).min().shift(1).values
    long_entry = (close > prev_hi) & ~np.isnan(prev_hi)
    long_exit = (close < prev_lo) & ~np.isnan(prev_lo)
    z = np.zeros(len(close), dtype=bool)
    cfg2 = FastSignalConfig(
        capital=cfg.capital,
        point_value=cfg.point_value,
        fee_per_side=cfg.fee_per_side,
        tax_rate=cfg.tax_rate,
        slip_ticks=cfg.slip_ticks,
        sessions=cfg.sessions,
        stop_points=cfg.stop_points,
        max_daily_loss=cfg.max_daily_loss,
        flatten_eod=cfg.flatten_eod,
        allow_short=False,
    )
    return _run_from_masks(
        bars,
        cfg2,
        long_entry=long_entry,
        short_entry=z,
        long_exit=long_exit,
        short_exit=z,
        reason_prefix="breakout_long",
    )


def _torb_entry_masks(
    bars: pd.DataFrame,
    probe_minutes: int,
    cfg: FastSignalConfig,
    *,
    once_per_day: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Timely Opening Range Breakout entries (no look-ahead).

    Per calendar day, within allowed sessions: first `probe_minutes` bars form
    OR high/low (those bars never enter). After probe, close > OR high → long;
    close < OR low → short. Range uses only completed probe bars.
    """
    if bars is None or bars.empty:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    df = bars.reset_index(drop=True)
    n = len(df)
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    ts = df["ts"].values if "ts" in df.columns else np.arange(n)
    if "session" in df.columns:
        sess = df["session"].astype(str).values
    else:
        sess = np.array(["day"] * n)
    allow = set(cfg.sessions) if cfg.sessions else {"day", "night"}
    in_sess = np.array([s in allow for s in sess], dtype=bool)

    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    probe = max(1, int(probe_minutes))

    i = 0
    while i < n:
        if not in_sess[i]:
            i += 1
            continue
        try:
            day = str(pd.Timestamp(ts[i]).date())
        except Exception:
            day = str(ts[i])[:10]
        j = i
        while j < n and in_sess[j]:
            try:
                dj = str(pd.Timestamp(ts[j]).date())
            except Exception:
                dj = str(ts[j])[:10]
            if dj != day:
                break
            j += 1
        # session block [i, j) for this calendar day
        length = j - i
        if length <= probe:
            i = j
            continue
        or_hi = float(np.max(high[i : i + probe]))
        or_lo = float(np.min(low[i : i + probe]))
        fired = False
        for k in range(i + probe, j):
            if once_per_day and fired:
                break
            c = float(close[k])
            if c > or_hi:
                long_entry[k] = True
                fired = True
                if once_per_day:
                    break
            elif cfg.allow_short and c < or_lo:
                short_entry[k] = True
                fired = True
                if once_per_day:
                    break
        i = j

    return long_entry, short_entry


def run_torb_fast(
    bars: pd.DataFrame,
    probe_minutes: int,
    cfg: FastSignalConfig,
    *,
    take_profit_points: float = 0.0,
    once_per_day: bool = True,
) -> BacktestResult:
    """
    TORB on 1m bars: probe opening range then breakout.
    Long-only uses optional fixed TP; short allowed only if cfg.allow_short.
    """
    long_entry, short_entry = _torb_entry_masks(
        bars, probe_minutes, cfg, once_per_day=once_per_day
    )
    if bars is None or bars.empty:
        return BacktestResult(final_equity=cfg.capital)

    tp = float(take_profit_points)
    if not cfg.allow_short:
        return _long_only_tp_loop(
            bars,
            cfg,
            long_entry,
            take_profit_points=tp,
            reason_entry=f"torb_p{probe_minutes}",
        )

    z = np.zeros(len(bars), dtype=bool)
    # both directions: no fixed TP in shared mask runner (stop + flatten only)
    return _run_from_masks(
        bars,
        cfg,
        long_entry=long_entry,
        short_entry=short_entry,
        long_exit=z,
        short_exit=z,
        reason_prefix=f"torb_p{probe_minutes}",
    )
