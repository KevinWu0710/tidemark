from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.types import Side, SignalDecision
from src.signals.base import BaseSignal
from src.signals.candidates import build_signal

_SIGNAL_KEYS = frozenset(
    {
        "inner",
        "lookback",
        "sessions",
        "timeframe",
        "rsi_period",
        "rsi_low",
        "rsi_high",
        "ema_period",
        "sma_fast",
        "sma_slow",
        "probe_mins",
        "cutoff_mins",
        "gap_th",
        "max_or_range",
        "vol_spike_mult",
        "allow_short",
        "flatten_off_session",
        "flatten_eod",
        "take_profit_points",  # 僅標記；實際停利在 strategy
    }
)


class TmfV1Signal(BaseSignal):
    """
    現役：日盤當沖 RSI 只做多（進場）+ 策略層固定停利／停損。

    掃描鎖定（日勝率>80%）：
    RSI28 reclaim>20 → 只做多；TP=40pt；stop=300；flatten 離日盤。
    rsi_high 設很高，出場交給 take_profit／stop／flatten。
    """

    DEFAULT_PARAMS = {
        "inner": "rsi_mr",
        "rsi_period": 28,
        "rsi_low": 20,
        "rsi_high": 99.5,
        "sessions": ["day"],
        "allow_short": False,
        "flatten_off_session": True,
        "timeframe": "1m",
    }

    def __init__(self, params: dict[str, Any] | None = None):
        raw = {**(params or {})}
        cleaned = {k: v for k, v in raw.items() if k in _SIGNAL_KEYS}
        cleaned.pop("take_profit_points", None)
        if cleaned.pop("flatten_eod", None):
            cleaned["flatten_off_session"] = True
        merged = {**self.DEFAULT_PARAMS, **cleaned}
        super().__init__(params=merged)
        inner_name = str(merged.get("inner", "rsi_mr"))
        inner_params = {
            "sessions": list(merged.get("sessions", ["day"])),
            "allow_short": bool(merged.get("allow_short", False)),
            "flatten_off_session": bool(merged.get("flatten_off_session", True)),
            "rsi_period": int(merged.get("rsi_period", 28)),
            "rsi_low": float(merged.get("rsi_low", 20)),
            "rsi_high": float(merged.get("rsi_high", 99.5)),
            "lookback": int(merged.get("lookback", 60)),
            "ema_period": int(merged.get("ema_period", 20)),
            "sma_fast": int(merged.get("sma_fast", 20)),
            "sma_slow": int(merged.get("sma_slow", 60)),
            "probe_mins": int(merged.get("probe_mins", 15)),
            "cutoff_mins": int(merged.get("cutoff_mins", 105)),
            "gap_th": float(merged.get("gap_th", 40.0)),
            "max_or_range": float(merged.get("max_or_range", 250.0)),
            "vol_spike_mult": float(merged.get("vol_spike_mult", 0.0)),
        }
        self._inner = build_signal(inner_name, inner_params)

    def evaluate(
        self,
        history: pd.DataFrame,
        position: Side,
        *,
        now_ts: Any = None,
    ) -> SignalDecision:
        return self._inner.evaluate(history, position, now_ts=now_ts)
