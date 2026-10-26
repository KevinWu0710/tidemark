"""
互補分流訊號（純價訊）：探針結束選 orb|rsi|flat，產出當日成交意圖。

不含口數／停損金額／券商；風控見 `src/strategy/complementary_risk.py`。
引擎實作仍在 `backtest/complementary_fast`（IS/OOS 同源），本模組是對外邊界。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtest.complementary_fast import (
    ComplementaryConfig,
    PROBE_END_MINUTES,
    run_complementary_fast,
    today_complementary_choice,
)
from src.core.types import SignalDecision, TradeAction
from src.signals.intents import TradeIntent


class ComplementarySignal:
    """
    bars → choice + intents（≤ as_of，無偷看）。
    """

    probe_end_minutes = PROBE_END_MINUTES

    def __init__(self, cfg: ComplementaryConfig | None = None):
        self.cfg = cfg or ComplementaryConfig.from_locked()

    def choice(
        self, bars: pd.DataFrame, *, as_of: str
    ) -> dict[str, Any]:
        return today_complementary_choice(bars, as_of=as_of, cfg=self.cfg)

    def intents(
        self,
        bars: pd.DataFrame,
        *,
        date: str,
        as_of_ts: Any,
    ) -> tuple[list[TradeIntent], dict[str, Any], dict[str, Any]]:
        """
        回傳 (intents, meta, choice_info)。
        choice 日必須 == date，否則 intents 空（禁止 fallback 前一日）。
        """
        as_of = pd.Timestamp(as_of_ts)
        choice_info = self.choice(bars, as_of=date)
        if not choice_info.get("ok") or str(choice_info.get("date")) != date:
            return [], {}, choice_info
        ch = choice_info.get("choice")
        if ch == "flat":
            return [], {}, choice_info

        result, meta = run_complementary_fast(bars, self.cfg)
        out: list[TradeIntent] = []
        for t in result.trades:
            t_ts = pd.Timestamp(t["ts"])
            if str(t_ts.date()) != date or t_ts > as_of:
                continue
            action_s = str(t.get("action", "")).upper()
            if action_s not in ("BUY", "SELL", "CLOSE"):
                continue
            out.append(
                TradeIntent(
                    ts=t_ts,
                    action=TradeAction(action_s),
                    price=float(t.get("price") or 0),
                    reason=str(t.get("reason") or f"complementary_{ch}"),
                    track=t.get("track") or ch,
                )
            )
        return out, meta, choice_info

    def evaluate(
        self,
        history: pd.DataFrame,
        position,  # Side — unused; complementary is batch-intent
        *,
        now_ts: Any = None,
        as_of_date: str | None = None,
    ) -> SignalDecision:
        """BaseSignal 風格入口：回傳當下建議（取最後一筆意圖或 HOLD）。"""
        date = as_of_date or (
            str(pd.Timestamp(now_ts).date()) if now_ts is not None else ""
        )
        if not date or history is None or history.empty:
            return SignalDecision(action=TradeAction.HOLD, reason="no_bars")
        intents, _meta, choice = self.intents(
            history, date=date, as_of_ts=now_ts or history.iloc[-1]["ts"]
        )
        if not choice.get("ok"):
            return SignalDecision(
                action=TradeAction.HOLD,
                reason="choice_fail",
                meta={"choice": choice},
            )
        if not intents:
            return SignalDecision(
                action=TradeAction.HOLD,
                reason="no_intent",
                meta={"choice": choice.get("choice")},
                ts=str(now_ts) if now_ts is not None else None,
            )
        last = intents[-1]
        return SignalDecision(
            action=last.action,
            reason=last.reason,
            confidence=1.0,
            meta={"choice": choice.get("choice"), "track": last.track, "n_intents": len(intents)},
            ts=str(last.ts),
        )
