from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.types import Side, SignalDecision, TradeAction
from src.data.sessions import label_session
from src.signals.base import BaseSignal


def _ts(val: Any) -> str | None:
    return str(val) if val is not None else None


class OpenGapOrbSignal(BaseSignal):
    """
    開盤 15m 區間突破 + 缺口延續 (Open 15m Range Breakout with Gap & Go Filter)。
    純因果訊號：時刻 t 僅看到 history[:t+1]。

    參數：
      - probe_mins: 開盤探針長度（預設 15 根 1分K，即 08:45–09:00）
      - cutoff_mins: 最晚進場時間（預設 105 分鐘，即 08:45+105m=10:30 後不追）
      - gap_th: 跳空門檻點數（預設 40.0 點）
      - max_or_range: 最大容許探針振幅（預設 250.0 點，超過視為過度宣洩不追）
      - allow_short: 是否允許做空（預設 False）
      - sessions: 允許開倉盤別（預設 ['day']）
      - flatten_off_session: 非允許盤別強制平倉（預設 True）
    """

    DEFAULT_PARAMS = {
        "probe_mins": 15,
        "cutoff_mins": 105,
        "gap_th": 40.0,
        "max_or_range": 250.0,
        "vol_spike_mult": 0.0,
        "allow_short": False,
        "sessions": ["day"],
        "flatten_off_session": True,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged)
        self.probe_mins = int(merged["probe_mins"])
        self.cutoff_mins = int(merged["cutoff_mins"])
        self.gap_th = float(merged["gap_th"])
        self.max_or_range = float(merged["max_or_range"])
        self.vol_spike_mult = float(merged["vol_spike_mult"])
        self.allow_short = bool(merged["allow_short"])
        self.sessions = list(merged["sessions"])
        self.flatten_off_session = bool(merged["flatten_off_session"])

    def evaluate(
        self,
        history: pd.DataFrame,
        position: Side,
        *,
        now_ts: Any = None,
    ) -> SignalDecision:
        if history is None or history.empty:
            return SignalDecision(TradeAction.HOLD, "empty_history", ts=_ts(now_ts))

        last_row = history.iloc[-1]
        ts_val = last_row.get("ts", now_ts)
        cur_sess = last_row.get("session")
        if cur_sess is None and ts_val is not None:
            cur_sess = label_session(pd.Timestamp(ts_val))

        # 非允許盤別處理
        if cur_sess not in self.sessions:
            if self.flatten_off_session and position != Side.FLAT:
                return SignalDecision(
                    TradeAction.CLOSE, "flatten_off_session", 0.9, ts=_ts(ts_val)
                )
            return SignalDecision(TradeAction.HOLD, "off_session", ts=_ts(ts_val))

        # 僅在日盤進行開盤突破分析
        if cur_sess != "day":
            return SignalDecision(TradeAction.HOLD, "non_day_session", ts=_ts(ts_val))

        cur_ts = pd.Timestamp(ts_val)
        cur_date_str = cur_ts.strftime("%Y-%m-%d")

        # 取得今日日盤所有已完成的 K 棒
        if "session" not in history.columns:
            sess_series = history["ts"].map(label_session)
        else:
            sess_series = history["session"]

        ts_series = pd.to_datetime(history["ts"])
        day_mask = (ts_series.dt.strftime("%Y-%m-%d") == cur_date_str) & (sess_series == "day")
        today_bars = history[day_mask]
        n_today = len(today_bars)

        # 探針期內不進場
        if n_today <= self.probe_mins:
            return SignalDecision(TradeAction.HOLD, "probing", ts=_ts(ts_val))

        # 超過 cutoff 不開新倉
        if position == Side.FLAT and n_today > self.cutoff_mins:
            return SignalDecision(TradeAction.HOLD, "past_entry_cutoff", ts=_ts(ts_val))

        # 計算探針高低點 (08:45 ~ 08:45 + probe_mins)
        probe_bars = today_bars.iloc[: self.probe_mins]
        or_high = float(probe_bars["high"].max())
        or_low = float(probe_bars["low"].min())
        or_range = or_high - or_low

        if self.max_or_range > 0 and or_range > self.max_or_range:
            return SignalDecision(TradeAction.HOLD, "or_range_too_wide", ts=_ts(ts_val))

        # 計算跳空缺口 (相對於前一日日盤收盤)
        today_open = float(probe_bars["open"].iloc[0])
        prior_day_bars = history[(sess_series == "day") & (ts_series < probe_bars["ts"].iloc[0])]
        prev_close = np.nan
        if len(prior_day_bars) > 0:
            prev_close = float(prior_day_bars["close"].iloc[-1])

        gap = (today_open - prev_close) if np.isfinite(prev_close) else 0.0

        # 缺口未補檢查
        gap_unfilled_long = (gap >= self.gap_th) and (np.isnan(prev_close) or or_low > prev_close)
        gap_unfilled_short = (gap <= -self.gap_th) and (np.isnan(prev_close) or or_high < prev_close)

        cur_close = float(last_row["close"])

        # 量能倍增檢查
        vol_ok = True
        if self.vol_spike_mult > 0 and "volume" in history.columns:
            avg_probe_vol = float(probe_bars["volume"].mean())
            cur_vol = float(last_row.get("volume", 0))
            vol_ok = (cur_vol >= self.vol_spike_mult * avg_probe_vol)

        # 進場判定
        if position == Side.FLAT:
            if gap_unfilled_long and cur_close > or_high and vol_ok:
                return SignalDecision(
                    TradeAction.BUY,
                    f"open_gap_orb long break {or_high:.1f} (gap={gap:+.0f})",
                    0.6,
                    ts=_ts(ts_val),
                    meta={"or_high": or_high, "or_low": or_low, "gap": gap},
                )
            if self.allow_short and gap_unfilled_short and cur_close < or_low and vol_ok:
                return SignalDecision(
                    TradeAction.SELL,
                    f"open_gap_orb short break {or_low:.1f} (gap={gap:+.0f})",
                    0.6,
                    ts=_ts(ts_val),
                    meta={"or_high": or_high, "or_low": or_low, "gap": gap},
                )

        return SignalDecision(TradeAction.HOLD, "hold", ts=_ts(ts_val))

