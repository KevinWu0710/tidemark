#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日盤事前過濾（strategy／director 層，不進 signals）。

現役用途：只擋 **日盤** 新多進場。夜盤目前 tmf_v1 不做單；
「夜盤大跌」只當隔夜風險代理，用來決定日盤早盤要不要觀望。

預設候選（research_long_prefilters 均衡過關）：
  前一夜盤報酬 ≤ -120 點 → 日盤開盤後 60 分鐘內不進
  且僅允許開盤後 ≤ 150 分鐘內進場（≈11:15 前）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.sessions import label_session

DAY_OPEN_MIN = 8 * 60 + 45  # 08:45


@dataclass
class DayPrefilterConfig:
    enabled: bool = False
    night_ret_le: float = -120.0
    skip_open_mins: int = 60
    max_entry_mins: int = 150  # 08:45 + 150m = 11:15

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> "DayPrefilterConfig":
        p = dict(params or {})
        return cls(
            enabled=bool(p.get("prefilter_enabled", False)),
            night_ret_le=float(p.get("prefilter_night_ret_le", -120.0)),
            skip_open_mins=int(p.get("prefilter_skip_open_mins", 60)),
            max_entry_mins=int(p.get("prefilter_max_entry_mins", 150)),
        )


def mins_from_day_open(ts: Any) -> int | None:
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return None
    if label_session(t) != "day":
        return None
    return int(t.hour * 60 + t.minute - DAY_OPEN_MIN)


def prior_night_return(history: pd.DataFrame, now_ts: Any) -> float:
    """
    日盤當下：取「上一根日盤收盤之後 → 本根之前」的夜盤段報酬（無偷看）。
    無夜盤資料 → 0。
    """
    if history is None or history.empty or now_ts is None:
        return 0.0
    df = history
    if "ts" not in df.columns:
        return 0.0
    ts = pd.to_datetime(df["ts"])
    now = pd.Timestamp(now_ts)
    if "session" in df.columns:
        sess = df["session"].astype(str)
    else:
        sess = ts.map(label_session)

    # last completed day bar strictly before today 08:45
    day_mask = sess == "day"
    day_before = df.loc[day_mask & (ts < now.replace(hour=8, minute=45, second=0, microsecond=0))]
    if day_before.empty:
        # fallback: any day bar before now
        day_before = df.loc[day_mask & (ts < now)]
    if day_before.empty:
        prev_day_last = None
    else:
        prev_day_last = pd.Timestamp(day_before["ts"].iloc[-1])

    night_mask = sess == "night"
    if prev_day_last is None:
        nseg = df.loc[night_mask & (ts < now)]
    else:
        nseg = df.loc[night_mask & (ts > prev_day_last) & (ts < now)]
    if len(nseg) < 2:
        return 0.0
    return float(nseg["close"].astype(float).iloc[-1] - nseg["open"].astype(float).iloc[0])


def allow_day_entry(
    history: pd.DataFrame,
    now_ts: Any,
    cfg: DayPrefilterConfig,
) -> tuple[bool, str]:
    """
    是否允許日盤新進場。非日盤時段 → 不由此過濾（交還 sessions 規則）。
    """
    if not cfg.enabled:
        return True, ""
    mins = mins_from_day_open(now_ts)
    if mins is None:
        return True, ""  # 非日盤：本模組不管
    if cfg.max_entry_mins > 0 and mins > cfg.max_entry_mins:
        return False, f"prefilter: after_{cfg.max_entry_mins}m_from_open"
    if cfg.skip_open_mins > 0 and mins < cfg.skip_open_mins:
        nret = prior_night_return(history, now_ts)
        if nret <= cfg.night_ret_le:
            return (
                False,
                f"prefilter: night_ret={nret:.1f}<={cfg.night_ret_le}_skip_{cfg.skip_open_mins}m",
            )
    return True, ""


def entry_allow_mask(bars: pd.DataFrame, cfg: DayPrefilterConfig) -> np.ndarray:
    """
    向量化日盤進場允許 mask（與 allow_day_entry 同一規則，供快路徑）。
    非日盤 bar → True（不擋；實際進場仍受 sessions 限制）。
    """
    n = len(bars) if bars is not None else 0
    if n == 0:
        return np.array([], dtype=bool)
    if not cfg.enabled:
        return np.ones(n, dtype=bool)

    df = bars.reset_index(drop=True)
    ts = pd.to_datetime(df["ts"])
    if "session" in df.columns:
        sess = df["session"].astype(str).values
    else:
        sess = np.array([label_session(t) for t in ts], dtype=object)

    mins = (ts.dt.hour * 60 + ts.dt.minute - DAY_OPEN_MIN).to_numpy()
    dates = ts.dt.strftime("%Y-%m-%d").to_numpy()

    # per calendar date: night ret before that day's first day bar
    night_ret_by_date: dict[str, float] = {}
    day_dates = sorted({d for d, s in zip(dates, sess) if s == "day"})
    for d in day_dates:
        day_ix = [i for i in range(n) if dates[i] == d and sess[i] == "day"]
        if not day_ix:
            continue
        first_day_i = day_ix[0]
        # night bars after previous day last bar and before first day bar
        prev_day_last = None
        for j in range(first_day_i - 1, -1, -1):
            if sess[j] == "day":
                prev_day_last = j
                break
        lo = (prev_day_last + 1) if prev_day_last is not None else 0
        nseg = [
            i
            for i in range(lo, first_day_i)
            if sess[i] == "night"
        ]
        if len(nseg) >= 2:
            o0 = float(df["open"].iloc[nseg[0]])
            c1 = float(df["close"].iloc[nseg[-1]])
            night_ret_by_date[d] = c1 - o0
        else:
            night_ret_by_date[d] = 0.0

    allow = np.ones(n, dtype=bool)
    for i in range(n):
        if sess[i] != "day":
            continue
        m = int(mins[i])
        if cfg.max_entry_mins > 0 and m > cfg.max_entry_mins:
            allow[i] = False
            continue
        if cfg.skip_open_mins > 0 and m < cfg.skip_open_mins:
            nret = float(night_ret_by_date.get(str(dates[i]), 0.0))
            if nret <= cfg.night_ret_le:
                allow[i] = False
    return allow
