"""
互補策略風控：一口、停損／保本／TP、時間停、日結、當日已平倉禁再開。

不發明進場指標；進場意圖來自 `signals.complementary`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from src.core.types import TradeAction
from src.data.sessions import label_session
from src.signals.intents import TradeIntent


def minutes_of_day(ts: pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    return int(t.hour) * 60 + int(t.minute)


def is_session_end(ts: pd.Timestamp) -> bool:
    mins = minutes_of_day(ts)
    sess = label_session(ts)
    if sess == "day":
        return mins >= (13 * 60 + 45)
    if sess == "closed":
        return (13 * 60 + 45) <= mins < (15 * 60)
    if sess == "night":
        return mins <= (5 * 60) and mins >= (4 * 60 + 55)
    return False


def probe_ready(ts: pd.Timestamp, *, probe_end_minutes: int) -> bool:
    """日盤探針結束後才允許分流成交。"""
    if label_session(ts) != "day":
        return True
    return minutes_of_day(ts) >= probe_end_minutes


@dataclass(frozen=True)
class RiskExitIntent:
    """策略要求平倉（尚未下單）。"""

    reason: str
    mark_price: float
    track: str
    ts: Any


@dataclass
class ComplementaryRiskStrategy:
    """
    max_lots=1；tick 風控預設每 10 秒檢查（抗毛刺）。
    risk 鍵對齊 ComplementaryConfig.risk_dict()。
    """

    risk: dict[str, Any]
    max_lots: int = 1
    risk_check_seconds: int = 10
    orb_be_lock_pts: float = 5.0
    orb_time_stop_mins: int = 45

    entry_price: Optional[float] = None
    entry_ts: Optional[pd.Timestamp] = None
    max_price: Optional[float] = None
    track: Optional[str] = None
    last_risk_check_ts: Optional[pd.Timestamp] = None
    completed_days: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.max_lots = 1  # 硬鎖

    def in_position(self) -> bool:
        return self.entry_price is not None

    def note_entry(
        self,
        *,
        fill_price: float,
        ts: Any,
        track: str | None,
    ) -> None:
        self.entry_price = float(fill_price)
        self.entry_ts = pd.Timestamp(ts)
        self.max_price = float(fill_price)
        self.track = track
        self.last_risk_check_ts = pd.Timestamp(ts)

    def note_exit(self, *, date: str) -> None:
        self.completed_days.add(str(date))
        self.entry_price = None
        self.entry_ts = None
        self.max_price = None
        self.track = None
        self.last_risk_check_ts = None

    def allow_bar_intent(
        self,
        intent: TradeIntent,
        *,
        date: str,
        session_end: bool,
    ) -> bool:
        """過濾 1m 訊號意圖：已平倉日禁再開；盤中忽略假 flatten_eod。"""
        trade_date = str(pd.Timestamp(intent.ts).date())
        if trade_date != date:
            return False
        if trade_date in self.completed_days:
            return False
        if intent.reason == "flatten_eod" and not session_end:
            return False
        return True

    def check_tick(
        self,
        ts: Any,
        price: float,
        *,
        lots: int,
    ) -> RiskExitIntent | None:
        """
        持倉中每 risk_check_seconds 檢查一次。
        回傳平倉意圖或 None（不負責下單／滑價）。
        """
        if lots <= 0 or self.entry_price is None:
            return None

        t = pd.Timestamp(ts)
        px = float(price)
        if self.max_price is None or px > self.max_price:
            self.max_price = px

        if self.last_risk_check_ts is not None:
            elapsed = (t - self.last_risk_check_ts).total_seconds()
            if elapsed < float(self.risk_check_seconds or 10):
                return None
        self.last_risk_check_ts = t

        entry = float(self.entry_price)
        favorable = float(self.max_price or px) - entry
        track = self.track or "orb"
        r = self.risk

        if track == "orb":
            be_trig = float(r.get("orb_be_trig") or 50.0)
            stop_pts = float(r.get("orb_stop") or 50.0)
            tp_pts = float(r.get("orb_tp") or 200.0)
            be_active = be_trig > 0 and favorable >= be_trig
            effective_stop = (
                (entry + self.orb_be_lock_pts) if be_active else (entry - stop_pts)
            )
            effective_tp = entry + tp_pts
            time_stop_mins = int(self.orb_time_stop_mins)
        else:
            stop_pts = float(r.get("rsi_stop") or 300.0)
            tp_pts = float(r.get("rsi_tp") or 40.0)
            be_active = False
            effective_stop = entry - stop_pts
            effective_tp = entry + tp_pts
            time_stop_mins = 0

        # 停損／停利／時間停 → 最後日結
        exit_reason: str | None = None
        if px <= effective_stop:
            exit_reason = "breakeven_stop" if be_active else "stop_loss"
        elif px >= effective_tp:
            exit_reason = "take_profit"
        elif (
            time_stop_mins > 0
            and self.entry_ts is not None
            and ((t - self.entry_ts).total_seconds() / 60.0) >= time_stop_mins
        ):
            exit_reason = "time_stop"
        elif is_session_end(t):
            exit_reason = "flatten_eod"

        if not exit_reason:
            return None
        return RiskExitIntent(
            reason=exit_reason, mark_price=px, track=track, ts=t
        )

    def intent_mid(self, intent: TradeIntent, *, slip_ticks: float) -> float:
        """還原 mid 給 PaperBroker（意圖價已含研究滑價）。"""
        slip = float(slip_ticks)
        px = float(intent.price)
        if intent.action == TradeAction.BUY:
            return px - slip
        if intent.action == TradeAction.SELL:
            return px + slip
        return px + slip  # CLOSE long

    def close_mid(self, mark_price: float, *, slip_ticks: float) -> float:
        """多單平倉 mid：mark + slip → paper fill ≈ mark。"""
        return float(mark_price) + float(slip_ticks)
