from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.types import Side, SignalDecision, TradeAction
from src.signals.base import BaseSignal


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


class EmaMomentumSignal(BaseSignal):
    """Trend: long if close>EMA and EMA rising; short if opposite. Exit on cross back."""

    def evaluate(
        self, history: pd.DataFrame, position: Side, *, now_ts: Any = None
    ) -> SignalDecision:
        period = int(self.params.get("ema_period", 20))
        if history is None or len(history) < period + 2:
            return SignalDecision(TradeAction.HOLD, "warmup", ts=_ts(now_ts))

        close = history["close"].astype(float)
        ema = _ema(close, period)
        c, e, e_prev = float(close.iloc[-1]), float(ema.iloc[-1]), float(ema.iloc[-2])
        rising = e > e_prev
        falling = e < e_prev

        allow_short = bool(self.params.get("allow_short", True))
        if position == Side.FLAT:
            if c > e and rising:
                return SignalDecision(
                    TradeAction.BUY, f"ema_mom long close>{period}EMA rising", 0.6, ts=_ts(now_ts)
                )
            if allow_short and c < e and falling:
                return SignalDecision(
                    TradeAction.SELL, f"ema_mom short close<{period}EMA falling", 0.6, ts=_ts(now_ts)
                )
        elif position == Side.LONG and c < e:
            return SignalDecision(TradeAction.CLOSE, "ema_mom exit long", 0.5, ts=_ts(now_ts))
        elif position == Side.SHORT and c > e:
            return SignalDecision(TradeAction.CLOSE, "ema_mom exit short", 0.5, ts=_ts(now_ts))
        return SignalDecision(TradeAction.HOLD, "ema_mom hold", ts=_ts(now_ts))


class RsiMeanRevertSignal(BaseSignal):
    """Mean-revert: buy RSI oversold reclaim; sell overbought fail. Flat-only entries."""

    def evaluate(
        self, history: pd.DataFrame, position: Side, *, now_ts: Any = None
    ) -> SignalDecision:
        period = int(self.params.get("rsi_period", 6))
        low_th = float(self.params.get("rsi_low", 25))
        high_th = float(self.params.get("rsi_high", 75))
        allow_short = bool(self.params.get("allow_short", True))
        if history is None or len(history) < period + 3:
            return SignalDecision(TradeAction.HOLD, "warmup", ts=_ts(now_ts))

        rsi = _rsi(history["close"].astype(float), period)
        r0, r1 = float(rsi.iloc[-1]), float(rsi.iloc[-2])

        if position == Side.FLAT:
            if r1 < low_th <= r0:
                return SignalDecision(
                    TradeAction.BUY, f"rsi reclaim>{low_th}", 0.55, ts=_ts(now_ts)
                )
            if allow_short and r1 > high_th >= r0:
                return SignalDecision(
                    TradeAction.SELL, f"rsi fail<{high_th}", 0.55, ts=_ts(now_ts)
                )
        elif position == Side.LONG and r0 >= high_th:
            return SignalDecision(TradeAction.CLOSE, "rsi long take profit zone", 0.5, ts=_ts(now_ts))
        elif position == Side.SHORT and r0 <= low_th:
            return SignalDecision(TradeAction.CLOSE, "rsi short take profit zone", 0.5, ts=_ts(now_ts))
        return SignalDecision(TradeAction.HOLD, "rsi hold", ts=_ts(now_ts))


class BreakoutSignal(BaseSignal):
    """Donchian-style: break N-bar high/low with ATR stop exit via opposite break."""

    def evaluate(
        self, history: pd.DataFrame, position: Side, *, now_ts: Any = None
    ) -> SignalDecision:
        lookback = int(self.params.get("lookback", 20))
        if history is None or len(history) < lookback + 2:
            return SignalDecision(TradeAction.HOLD, "warmup", ts=_ts(now_ts))

        window = history.iloc[-(lookback + 1) : -1]
        hi = float(window["high"].max())
        lo = float(window["low"].min())
        c = float(history["close"].iloc[-1])

        allow_short = bool(self.params.get("allow_short", True))
        if position == Side.FLAT:
            if c > hi:
                return SignalDecision(TradeAction.BUY, f"breakout>{lookback}high", 0.65, ts=_ts(now_ts))
            if allow_short and c < lo:
                return SignalDecision(TradeAction.SELL, f"breakout<{lookback}low", 0.65, ts=_ts(now_ts))
        elif position == Side.LONG and c < lo:
            return SignalDecision(TradeAction.CLOSE, "breakout exit long", 0.5, ts=_ts(now_ts))
        elif position == Side.SHORT and c > hi:
            return SignalDecision(TradeAction.CLOSE, "breakout exit short", 0.5, ts=_ts(now_ts))
        return SignalDecision(TradeAction.HOLD, "breakout hold", ts=_ts(now_ts))


class SessionFilterSignal(BaseSignal):
    """Wrapper: only allow inner signal during allowed sessions (day/night/both)."""

    def __init__(self, inner: BaseSignal, sessions: list[str] | None = None, params: dict | None = None):
        super().__init__(params=params)
        self.inner = inner
        self.sessions = set(sessions or ["day", "night"])

    def evaluate(
        self, history: pd.DataFrame, position: Side, *, now_ts: Any = None
    ) -> SignalDecision:
        if history is None or history.empty:
            return SignalDecision(TradeAction.HOLD, "empty", ts=_ts(now_ts))
        sess = str(history.iloc[-1].get("session", "day"))
        if sess not in self.sessions:
            # still allow close outside session to flatten risk at session edge
            if position != Side.FLAT and self.params.get("flatten_off_session"):
                return SignalDecision(
                    TradeAction.CLOSE, f"flatten off-session ({sess})", 0.4, ts=_ts(now_ts)
                )
            return SignalDecision(TradeAction.HOLD, f"skip session={sess}", ts=_ts(now_ts))
        return self.inner.evaluate(history, position, now_ts=now_ts)


def build_signal(name: str, params: dict[str, Any] | None = None) -> BaseSignal:
    params = params or {}
    sessions = params.get("sessions", ["day", "night"])
    if isinstance(sessions, str):
        sessions = [sessions]

    if name == "ema_mom":
        inner: BaseSignal = EmaMomentumSignal(params)
    elif name == "rsi_mr":
        inner = RsiMeanRevertSignal(params)
    elif name == "breakout":
        inner = BreakoutSignal(params)
    elif name == "breakout_long":
        inner = BreakoutSignal({**params, "allow_short": False})
    elif name == "open_gap_orb":
        from src.signals.open_gap_orb import OpenGapOrbSignal
        return OpenGapOrbSignal(params)
    else:
        raise ValueError(f"unknown signal: {name}")

    return SessionFilterSignal(
        inner,
        sessions=list(sessions),
        params={"flatten_off_session": bool(params.get("flatten_off_session", False))},
    )


def _ts(now_ts: Any) -> str | None:
    return str(now_ts) if now_ts is not None else None
