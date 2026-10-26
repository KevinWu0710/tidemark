from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.core.types import Side, SignalDecision, TradeAction
from src.signals.base import BaseSignal
from src.signals.tmf_v1 import TmfV1Signal
from src.strategy.day_prefilters import DayPrefilterConfig, allow_day_entry

_STRATEGY_KEYS = frozenset(
    {
        "stop_points",
        "take_profit_points",
        "time_stop_bars",
        "breakeven_points",
        "max_daily_loss",
        "max_lots",
        "capital",
        "prefilter_enabled",
        "prefilter_night_ret_le",
        "prefilter_skip_open_mins",
        "prefilter_max_entry_mins",
    }
)


class TmfSessionStrategy:
    """
    策略模組（獨立）：口數、停損、停利、保本、時間停損、單日虧損 — 不發明買賣點。
    """

    def __init__(
        self,
        signal: BaseSignal | None = None,
        *,
        max_lots: int = 1,
        capital: float = 10000,
        params: dict[str, Any] | None = None,
        stop_points: float = 0,
        take_profit_points: float = 0,
        time_stop_bars: int = 0,
        breakeven_points: float = 0,
        max_daily_loss: float = 0,
    ):
        params = dict(params or {})
        signal_params = {
            k: v for k, v in params.items() if k not in _STRATEGY_KEYS
        }
        self.signal = signal or TmfV1Signal(params=signal_params)
        self.max_lots = int(params.get("max_lots", max_lots) or max_lots)
        self.capital = float(params.get("capital", capital) or capital)
        self.stop_points = float(params.get("stop_points", stop_points) or 0)
        self.take_profit_points = float(
            params.get("take_profit_points", take_profit_points) or 0
        )
        self.time_stop_bars = int(params.get("time_stop_bars", time_stop_bars) or 0)
        self.breakeven_points = float(params.get("breakeven_points", breakeven_points) or 0)
        self.max_daily_loss = float(params.get("max_daily_loss", max_daily_loss) or 0)
        self.prefilter = DayPrefilterConfig.from_params(params)
        self.position: Side = Side.FLAT
        self.lots: int = 0
        self.entry_price: Optional[float] = None
        self.bars_held: int = 0
        self.highest_price: float = 0.0
        self.lowest_price: float = 0.0
        self._day_key: Optional[str] = None
        self._day_realized: float = 0.0
        self._history_ref: pd.DataFrame | None = None

    def on_bar(self, history: pd.DataFrame, *, now_ts: Any = None) -> SignalDecision:
        self._roll_day(now_ts)
        self._history_ref = history
        if history is not None and not history.empty and self.lots > 0 and self.entry_price:
            self.bars_held += 1
            if self.position == Side.LONG:
                self.highest_price = max(self.highest_price, float(history.iloc[-1].get("high", history.iloc[-1]["close"])))
            elif self.position == Side.SHORT:
                self.lowest_price = min(self.lowest_price, float(history.iloc[-1].get("low", history.iloc[-1]["close"])))
            px = float(history.iloc[-1]["close"])
            risk_dec = self._check_stop_or_tp(px, now_ts)
            if risk_dec is not None:
                return risk_dec

        raw = self.signal.evaluate(history, self.position, now_ts=now_ts)
        return self._apply_risk(raw, now_ts=now_ts, history=history)

    def _roll_day(self, now_ts: Any) -> None:
        key = str(now_ts)[:10] if now_ts is not None else None
        if key != self._day_key:
            self._day_key = key
            self._day_realized = 0.0

    def _check_stop_or_tp(self, px: float, now_ts: Any) -> SignalDecision | None:
        if self.entry_price is None:
            return None
        ts = str(now_ts) if now_ts is not None else None
        if self.time_stop_bars > 0 and self.bars_held >= self.time_stop_bars:
            return SignalDecision(
                TradeAction.CLOSE,
                f"time_stop_{self.time_stop_bars}bars",
                0.9,
                ts=ts,
                meta={"layer": "strategy"},
            )
        if self.position == Side.LONG:
            effective_sl = (self.entry_price - self.stop_points) if self.stop_points > 0 else -1e9
            if self.breakeven_points > 0 and (self.highest_price - self.entry_price) >= self.breakeven_points:
                effective_sl = max(effective_sl, self.entry_price + 5.0)
            if self.stop_points > 0 and px <= effective_sl:
                reason = f"breakeven_{self.breakeven_points}pt" if effective_sl > (self.entry_price - self.stop_points) else f"stop_loss_{self.stop_points}pt"
                return SignalDecision(
                    TradeAction.CLOSE,
                    reason,
                    0.9,
                    ts=ts,
                    meta={"layer": "strategy"},
                )
            if self.take_profit_points > 0 and (px - self.entry_price) >= self.take_profit_points:
                return SignalDecision(
                    TradeAction.CLOSE,
                    f"take_profit_{self.take_profit_points}pt",
                    0.9,
                    ts=ts,
                    meta={"layer": "strategy"},
                )
        if self.position == Side.SHORT:
            effective_sl = (self.entry_price + self.stop_points) if self.stop_points > 0 else 1e9
            if self.breakeven_points > 0 and (self.entry_price - self.lowest_price) >= self.breakeven_points:
                effective_sl = min(effective_sl, self.entry_price - 5.0)
            if self.stop_points > 0 and px >= effective_sl:
                reason = f"breakeven_{self.breakeven_points}pt" if effective_sl < (self.entry_price + self.stop_points) else f"stop_loss_{self.stop_points}pt"
                return SignalDecision(
                    TradeAction.CLOSE,
                    reason,
                    0.9,
                    ts=ts,
                    meta={"layer": "strategy"},
                )
            if self.take_profit_points > 0 and (self.entry_price - px) >= self.take_profit_points:
                return SignalDecision(
                    TradeAction.CLOSE,
                    f"take_profit_{self.take_profit_points}pt",
                    0.9,
                    ts=ts,
                    meta={"layer": "strategy"},
                )
        return None

    def _apply_risk(
        self,
        decision: SignalDecision,
        *,
        now_ts: Any = None,
        history: pd.DataFrame | None = None,
    ) -> SignalDecision:
        if decision.action in (TradeAction.BUY, TradeAction.SELL):
            if self.lots >= self.max_lots:
                return SignalDecision(
                    action=TradeAction.HOLD,
                    reason=f"blocked: already at max_lots={self.max_lots}",
                    confidence=decision.confidence,
                    ts=decision.ts,
                    meta={**decision.meta, "blocked": True, "layer": "strategy"},
                )
            if self.max_lots < 1:
                return SignalDecision(
                    action=TradeAction.HOLD,
                    reason="blocked: max_lots < 1",
                    confidence=0.0,
                    ts=decision.ts,
                    meta={"layer": "strategy"},
                )
            if self.max_daily_loss > 0 and self._day_realized <= -abs(self.max_daily_loss):
                return SignalDecision(
                    action=TradeAction.HOLD,
                    reason=f"blocked: max_daily_loss={self.max_daily_loss}",
                    confidence=0.0,
                    ts=decision.ts or (str(now_ts) if now_ts else None),
                    meta={"blocked": True, "layer": "strategy"},
                )
            # 日盤事前過濾：只擋新進場（BUY）；現役不做空，SELL 進場也擋以備之後
            if decision.action == TradeAction.BUY or (
                decision.action == TradeAction.SELL and self.position == Side.FLAT
            ):
                hist = history if history is not None else self._history_ref
                ok, why = allow_day_entry(hist, now_ts, self.prefilter)
                if not ok:
                    return SignalDecision(
                        action=TradeAction.HOLD,
                        reason=f"blocked: {why}",
                        confidence=0.0,
                        ts=decision.ts or (str(now_ts) if now_ts else None),
                        meta={
                            **decision.meta,
                            "blocked": True,
                            "layer": "strategy_prefilter",
                        },
                    )
        meta = {**decision.meta, "signal_reason": decision.reason}
        return SignalDecision(
            action=decision.action,
            reason=decision.reason,
            confidence=decision.confidence,
            ts=decision.ts,
            meta=meta,
        )

    def apply_fill(
        self,
        action: TradeAction,
        lots: int = 1,
        price: float | None = None,
        pnl: float | None = None,
    ) -> None:
        lots = min(lots, self.max_lots)
        if action == TradeAction.BUY:
            self.position = Side.LONG
            self.lots = lots
            self.bars_held = 0
            if price is not None:
                self.entry_price = float(price)
                self.highest_price = float(price)
                self.lowest_price = float(price)
        elif action == TradeAction.SELL:
            self.position = Side.SHORT
            self.lots = lots
            self.bars_held = 0
            if price is not None:
                self.entry_price = float(price)
                self.highest_price = float(price)
                self.lowest_price = float(price)
        elif action == TradeAction.CLOSE:
            self.position = Side.FLAT
            self.lots = 0
            self.entry_price = None
            self.bars_held = 0
            self.highest_price = 0.0
            self.lowest_price = 0.0
            if pnl is not None:
                self._day_realized += float(pnl)
